import json
import math
import random
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from handcrafted_features import FEATURE_PRESETS, FEATURE_REGISTRY
from models import build_model
from pair_features import (
    apply_feature_scaler,
    build_node_feature_mapping,
    fit_feature_scaler,
    load_pair_arrays,
)
from train import (
    build_pair_datasets,
    evaluate_pair_mlp,
    make_pair_loader,
    train_pair_mlp_step,
)


def sample_log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def choose(rng, values):
    values = list(values)
    return values[rng.randrange(len(values))]


def resolve_search_candidate_features(cfg):
    configured = list(cfg.search.get("candidate_features", []))
    if configured:
        unknown = [name for name in configured if name not in FEATURE_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown search.candidate_features: {unknown}")
        return configured
    return list(FEATURE_REGISTRY.keys())


def normalize_feature_names(feature_names):
    return list(dict.fromkeys(feature_names))


def sample_feature_spec(cfg, rng):
    mode = cfg.search.get("feature_selection_mode", "presets")
    mandatory = normalize_feature_names(cfg.search.get("mandatory_features", []))
    candidates = [name for name in resolve_search_candidate_features(cfg) if name not in mandatory]

    if mode == "presets":
        preset = choose(rng, cfg.search.presets)
        return {
            "kind": "preset",
            "preset": preset,
            "feature_names": list(FEATURE_PRESETS[preset]),
        }

    if mode == "subsets":
        subset_size = rng.randint(
            max(int(cfg.search.subset_min_size), len(mandatory)),
            max(int(cfg.search.subset_max_size), len(mandatory)),
        )
        subset_size = min(subset_size, len(mandatory) + len(candidates))
        extra_k = max(0, subset_size - len(mandatory))
        sampled = rng.sample(candidates, extra_k) if extra_k > 0 else []
        feature_names = normalize_feature_names(mandatory + sampled)
        return {
            "kind": "subset",
            "preset": None,
            "feature_names": feature_names,
        }

    if mode == "mixed":
        if rng.random() < float(cfg.search.get("preset_probability", 0.4)):
            preset = choose(rng, cfg.search.presets)
            return {
                "kind": "preset",
                "preset": preset,
                "feature_names": list(FEATURE_PRESETS[preset]),
            }
        subset_size = rng.randint(
            max(int(cfg.search.subset_min_size), len(mandatory)),
            max(int(cfg.search.subset_max_size), len(mandatory)),
        )
        subset_size = min(subset_size, len(mandatory) + len(candidates))
        extra_k = max(0, subset_size - len(mandatory))
        sampled = rng.sample(candidates, extra_k) if extra_k > 0 else []
        feature_names = normalize_feature_names(mandatory + sampled)
        return {
            "kind": "subset",
            "preset": None,
            "feature_names": feature_names,
        }

    raise ValueError(f"Unknown search.feature_selection_mode: {mode}")


def feature_spec_key(feature_spec):
    if feature_spec["kind"] == "preset":
        return ("preset", feature_spec["preset"])
    return ("subset", tuple(feature_spec["feature_names"]))


def build_trial_cfg(base_cfg, trial_params):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.model.kind = "pair_mlp"
    cfg.pair_mlp.feature_builder = "handcrafted"
    cfg.pair_mlp.notebook_compat = True
    if trial_params["feature_kind"] == "preset":
        cfg.pair_mlp.handcrafted_preset = trial_params["preset"]
        cfg.pair_mlp.handcrafted_selected_features = []
    else:
        cfg.pair_mlp.handcrafted_preset = "baseline_small"
        cfg.pair_mlp.handcrafted_selected_features = list(trial_params["feature_names"])
    cfg.pair_mlp.handcrafted_proj_dim = trial_params["handcrafted_proj_dim"]
    cfg.pair_mlp.hidden_dim_1 = trial_params["hidden_dim_1"]
    cfg.pair_mlp.hidden_dim_2 = trial_params["hidden_dim_2"]
    cfg.pair_mlp.dropout_1 = trial_params["dropout_1"]
    cfg.pair_mlp.dropout_2 = trial_params["dropout_2"]
    cfg.pair_mlp.dropout_3 = trial_params["dropout_3"]
    cfg.pair_mlp.handcrafted_lr = trial_params["handcrafted_lr"]
    cfg.pair_mlp.handcrafted_wd = trial_params["handcrafted_wd"]
    cfg.pair_mlp.batch_size = trial_params["batch_size"]
    cfg.pair_mlp.handcrafted_patience = trial_params["handcrafted_patience"]
    cfg.pair_mlp.handcrafted_lr_patience = trial_params["handcrafted_lr_patience"]
    cfg.pair_mlp.handcrafted_epochs = trial_params["handcrafted_epochs"]
    cfg.pair_mlp.handcrafted_eval_every = 1
    cfg.pair_mlp.handcrafted_use_scheduler = True
    return cfg


def sample_trial_params(cfg, rng):
    search_cfg = cfg.search
    feature_spec = sample_feature_spec(cfg, rng)
    return {
        "feature_kind": feature_spec["kind"],
        "preset": feature_spec["preset"],
        "feature_names": feature_spec["feature_names"],
        "handcrafted_proj_dim": choose(rng, search_cfg.proj_dims),
        "hidden_dim_1": choose(rng, search_cfg.hidden_dim_1),
        "hidden_dim_2": choose(rng, search_cfg.hidden_dim_2),
        "dropout_1": choose(rng, search_cfg.dropout_1),
        "dropout_2": choose(rng, search_cfg.dropout_2),
        "dropout_3": choose(rng, search_cfg.dropout_3),
        "handcrafted_lr": sample_log_uniform(rng, search_cfg.lr_min, search_cfg.lr_max),
        "handcrafted_wd": sample_log_uniform(
            rng,
            max(search_cfg.wd_min, 1e-8),
            max(search_cfg.wd_max, 1e-8),
        )
        if search_cfg.wd_max > 0
        else 0.0,
        "batch_size": choose(rng, search_cfg.batch_sizes),
        "handcrafted_patience": choose(rng, search_cfg.patience),
        "handcrafted_lr_patience": choose(rng, search_cfg.lr_patience),
        "handcrafted_epochs": max(choose(rng, search_cfg.patience) * 4, 60),
    }


def make_dataset_cache(cfg, trial_params_list):
    cache = {}
    candidate_specs = []
    for trial_params in trial_params_list:
        candidate_specs.append(
            {
                "kind": trial_params["feature_kind"],
                "preset": trial_params["preset"],
                "feature_names": list(trial_params["feature_names"]),
            }
        )

    for feature_spec in candidate_specs:
        cache_key = feature_spec_key(feature_spec)
        if cache_key in cache:
            continue

        preset_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        preset_cfg.model.kind = "pair_mlp"
        preset_cfg.pair_mlp.feature_builder = "handcrafted"
        preset_cfg.pair_mlp.notebook_compat = True
        if feature_spec["kind"] == "preset":
            preset_cfg.pair_mlp.handcrafted_preset = feature_spec["preset"]
            preset_cfg.pair_mlp.handcrafted_selected_features = []
        else:
            preset_cfg.pair_mlp.handcrafted_preset = "baseline_small"
            preset_cfg.pair_mlp.handcrafted_selected_features = list(feature_spec["feature_names"])
        node_array, train_array = load_pair_arrays(preset_cfg, include_test=False)
        node_features, remapping = build_node_feature_mapping(node_array, preset_cfg)
        pairs = train_array[:, :2].astype(np.int64)
        labels = train_array[:, 2].astype(np.float32)

        datasets = build_pair_datasets(preset_cfg, node_features, remapping, pairs, labels)
        x_train_s, scaler_stats = fit_feature_scaler(datasets["x_train"])
        x_val_s = apply_feature_scaler(datasets["x_val"], scaler_stats)
        x_test_s = apply_feature_scaler(datasets["x_test"], scaler_stats)

        cache[cache_key] = {
            "x_train": x_train_s,
            "y_train": datasets["y_train"],
            "x_val": x_val_s,
            "y_val": datasets["y_val"],
            "x_test": x_test_s,
            "y_test": datasets["y_test"],
            "input_dim": int(x_train_s.shape[1]),
            "feature_names": list(feature_spec["feature_names"]),
            "feature_kind": feature_spec["kind"],
            "preset": feature_spec["preset"],
        }

    return cache


def run_trial(trial_index, trial_cfg, trial_params, dataset_bundle, device):
    model = build_model(trial_cfg, dataset_bundle["input_dim"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(trial_cfg.pair_mlp.handcrafted_lr),
        weight_decay=float(trial_cfg.pair_mlp.handcrafted_wd),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(trial_cfg.pair_mlp.handcrafted_lr_decay_factor),
        patience=int(trial_cfg.pair_mlp.handcrafted_lr_patience),
        min_lr=float(trial_cfg.pair_mlp.handcrafted_min_lr),
    )
    criterion = nn.BCEWithLogitsLoss()

    train_loader = make_pair_loader(
        dataset_bundle["x_train"],
        dataset_bundle["y_train"],
        batch_size=int(trial_cfg.pair_mlp.batch_size),
        shuffle=True,
    )
    val_loader = make_pair_loader(
        dataset_bundle["x_val"],
        dataset_bundle["y_val"],
        batch_size=int(trial_cfg.pair_mlp.batch_size),
        shuffle=False,
    )
    test_loader = make_pair_loader(
        dataset_bundle["x_test"],
        dataset_bundle["y_test"],
        batch_size=int(trial_cfg.pair_mlp.batch_size),
        shuffle=False,
    )

    best_state = None
    best_epoch = 0
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    wait = 0
    max_epochs = int(trial_cfg.pair_mlp.handcrafted_epochs)
    patience = int(trial_cfg.pair_mlp.handcrafted_patience)

    for epoch in range(1, max_epochs + 1):
        running_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            batch_loss = train_pair_mlp_step(model, optimizer, xb, yb, criterion)
            running_loss += batch_loss * xb.size(0)
            total_count += xb.size(0)

        train_loss = running_loss / max(total_count, 1)
        val_loss, val_acc, val_auc, val_ap = evaluate_pair_mlp(val_loader, model, criterion, device)
        scheduler.step(val_auc)

        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, val_auc, val_ap = evaluate_pair_mlp(val_loader, model, criterion, device)
    test_loss, test_acc, test_auc, test_ap = evaluate_pair_mlp(test_loader, model, criterion, device)

    return {
        "trial": trial_index,
        "best_epoch": best_epoch,
        "train_loss_last": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_ap": val_ap,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "params": {
            "feature_kind": trial_params["feature_kind"],
            "preset": trial_params["preset"],
            "feature_names": list(trial_params["feature_names"]),
            "handcrafted_proj_dim": int(trial_cfg.pair_mlp.handcrafted_proj_dim),
            "hidden_dim_1": int(trial_cfg.pair_mlp.hidden_dim_1),
            "hidden_dim_2": int(trial_cfg.pair_mlp.hidden_dim_2),
            "dropout_1": float(trial_cfg.pair_mlp.dropout_1),
            "dropout_2": float(trial_cfg.pair_mlp.dropout_2),
            "dropout_3": float(trial_cfg.pair_mlp.dropout_3),
            "handcrafted_lr": float(trial_cfg.pair_mlp.handcrafted_lr),
            "handcrafted_wd": float(trial_cfg.pair_mlp.handcrafted_wd),
            "batch_size": int(trial_cfg.pair_mlp.batch_size),
            "handcrafted_patience": int(trial_cfg.pair_mlp.handcrafted_patience),
            "handcrafted_lr_patience": int(trial_cfg.pair_mlp.handcrafted_lr_patience),
            "handcrafted_epochs": int(trial_cfg.pair_mlp.handcrafted_epochs),
        },
    }


def save_results(cfg, results):
    output_dir = Path(cfg.search.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda item: item["val_auc"], reverse=True)
    results_path = output_dir / "handcrafted_search_results.json"
    results_path.write_text(json.dumps(sorted_results, indent=2))

    top_k = int(cfg.search.top_k)
    leaderboard_path = output_dir / "handcrafted_search_top.txt"
    lines = []
    for item in sorted_results[:top_k]:
        params = item["params"]
        feature_desc = params["preset"]
        if params["feature_names"]:
            feature_desc = f"subset[{len(params['feature_names'])}]"
        lines.append(
            f"trial={item['trial']} val_auc={item['val_auc']:.4f} test_auc={item['test_auc']:.4f} "
            f"features={feature_desc} proj={params['handcrafted_proj_dim']} "
            f"h1={params['hidden_dim_1']} h2={params['hidden_dim_2']} "
            f"lr={params['handcrafted_lr']:.5g} wd={params['handcrafted_wd']:.5g}"
        )
    leaderboard_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    if sorted_results:
        best = sorted_results[0]["params"]
        overrides = [
            "model.kind=pair_mlp",
            "pair_mlp.feature_builder=handcrafted",
            "pair_mlp.notebook_compat=true",
            f"pair_mlp.handcrafted_proj_dim={best['handcrafted_proj_dim']}",
            f"pair_mlp.hidden_dim_1={best['hidden_dim_1']}",
            f"pair_mlp.hidden_dim_2={best['hidden_dim_2']}",
            f"pair_mlp.dropout_1={best['dropout_1']}",
            f"pair_mlp.dropout_2={best['dropout_2']}",
            f"pair_mlp.dropout_3={best['dropout_3']}",
            f"pair_mlp.handcrafted_lr={best['handcrafted_lr']}",
            f"pair_mlp.handcrafted_wd={best['handcrafted_wd']}",
            f"pair_mlp.batch_size={best['batch_size']}",
            f"pair_mlp.handcrafted_patience={best['handcrafted_patience']}",
            f"pair_mlp.handcrafted_lr_patience={best['handcrafted_lr_patience']}",
            f"pair_mlp.handcrafted_epochs={best['handcrafted_epochs']}",
        ]
        if best["feature_names"]:
            feature_list = ",".join(best["feature_names"])
            overrides.append(f"'pair_mlp.handcrafted_selected_features=[{feature_list}]'")
        else:
            overrides.append(f"pair_mlp.handcrafted_preset={best['preset']}")
        (output_dir / "handcrafted_best_overrides.txt").write_text(" ".join(overrides) + "\n")

    return output_dir


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    rng = random.Random(int(cfg.search.seed))
    np.random.seed(int(cfg.search.seed))
    torch.manual_seed(int(cfg.search.seed))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    trial_params_list = [sample_trial_params(cfg, rng) for _ in range(int(cfg.search.n_trials))]
    dataset_cache = make_dataset_cache(cfg, trial_params_list)

    results = []
    for trial_index, trial_params in enumerate(trial_params_list, start=1):
        trial_cfg = build_trial_cfg(cfg, trial_params)
        dataset_bundle = dataset_cache[feature_spec_key(
            {
                "kind": trial_params["feature_kind"],
                "preset": trial_params["preset"],
                "feature_names": trial_params["feature_names"],
            }
        )]
        result = run_trial(trial_index, trial_cfg, trial_params, dataset_bundle, device)
        results.append(result)
        feature_desc = trial_params["preset"] or f"subset[{len(trial_params['feature_names'])}]"
        print(
            f"[trial {trial_index:02d}] val_auc={result['val_auc']:.4f} "
            f"test_auc={result['test_auc']:.4f} features={feature_desc} "
            f"proj={result['params']['handcrafted_proj_dim']} "
            f"lr={result['params']['handcrafted_lr']:.5g}"
        )

    output_dir = save_results(cfg, results)
    best = max(results, key=lambda item: item["val_auc"])
    print(f"Best trial: {best['trial']} | val_auc={best['val_auc']:.4f} | test_auc={best['test_auc']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
