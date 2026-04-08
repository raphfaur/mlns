import subprocess

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

from handcrafted_gnn import build_handcrafted_gnn_test_data
from handcrafted_features import (
    build_handcrafted_context,
    build_handcrafted_feature_matrix_with_context,
    resolve_handcrafted_feature_names,
)
from make_dataset import build_node_features, resolve_data_path
from models import build_model
from pair_features import (
    apply_feature_scaler,
    build_node_feature_mapping,
    build_pair_feature_matrix,
    build_positive_graph,
    compute_centrality_maps,
    fit_feature_scaler,
    load_pair_arrays,
)


def infer_pair_mlp_model_config(checkpoint):
    model_cfg = checkpoint.get("pair_mlp_model_config")
    if model_cfg is not None:
        return model_cfg

    state_dict = checkpoint["model_state_dict"]
    return {
        "proj_dim": int(state_dict["net.0.weight"].shape[0]),
        "handcrafted_proj_dim": int(state_dict["net.0.weight"].shape[0]),
        "hidden_dim_1": int(state_dict["net.4.weight"].shape[0]),
        "hidden_dim_2": int(state_dict["net.7.weight"].shape[0]),
        "dropout_1": 0.30,
        "dropout_2": 0.25,
        "dropout_3": 0.15,
        "batch_size": 256,
    }


def restore_pair_mlp_cfg_from_checkpoint(cfg, checkpoint):
    model_cfg = infer_pair_mlp_model_config(checkpoint)
    cfg.pair_mlp.proj_dim = int(model_cfg.get("proj_dim", cfg.pair_mlp.get("proj_dim", 64)))
    cfg.pair_mlp.handcrafted_proj_dim = int(
        model_cfg.get(
            "handcrafted_proj_dim",
            cfg.pair_mlp.get("handcrafted_proj_dim", cfg.pair_mlp.get("proj_dim", 64)),
        )
    )
    cfg.pair_mlp.hidden_dim_1 = int(
        model_cfg.get("hidden_dim_1", cfg.pair_mlp.get("hidden_dim_1", 64))
    )
    cfg.pair_mlp.hidden_dim_2 = int(
        model_cfg.get("hidden_dim_2", cfg.pair_mlp.get("hidden_dim_2", 32))
    )
    cfg.pair_mlp.dropout_1 = float(
        model_cfg.get("dropout_1", cfg.pair_mlp.get("dropout_1", 0.30))
    )
    cfg.pair_mlp.dropout_2 = float(
        model_cfg.get("dropout_2", cfg.pair_mlp.get("dropout_2", 0.25))
    )
    cfg.pair_mlp.dropout_3 = float(
        model_cfg.get("dropout_3", cfg.pair_mlp.get("dropout_3", 0.15))
    )
    cfg.pair_mlp.batch_size = int(
        model_cfg.get("batch_size", cfg.pair_mlp.get("batch_size", 256))
    )


def build_gnn_test_data(cfg):
    data_node = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "node_information.csv"),
        has_header=False,
    )
    train_df = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "train.txt"),
        separator=" ",
        has_header=False,
        new_columns=["a", "b", "label"],
    )
    test_df = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "test.txt"),
        separator=" ",
        has_header=False,
        new_columns=["a", "b"],
    )
    node_array = data_node.to_numpy()
    train_array = train_df.to_numpy()
    test_array = test_df.to_numpy()

    x_gnn, remapping = build_node_features(node_array, cfg)

    train_edges = train_array[:, :2]
    train_labels = train_array[:, 2]
    sender = []
    receiver = []

    for edge, label in zip(train_edges, train_labels):
        if label != 1:
            continue
        node_i, node_j = edge[0], edge[1]
        if node_i not in remapping or node_j not in remapping:
            continue
        sender.append(remapping[node_i])
        receiver.append(remapping[node_j])

    edge_index = torch.tensor([sender, receiver], dtype=torch.long)

    test_sender = []
    test_receiver = []
    valid_mask = []
    for edge in test_array:
        node_i, node_j = edge[0], edge[1]
        if node_i in remapping and node_j in remapping:
            test_sender.append(remapping[node_i])
            test_receiver.append(remapping[node_j])
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    test_edge_label_index = torch.tensor([test_sender, test_receiver], dtype=torch.long)
    test_data = Data(x=x_gnn, edge_index=edge_index, edge_label_index=test_edge_label_index)
    return test_data, test_df, np.array(valid_mask, dtype=bool)


def restore_handcrafted_gnn_cfg_from_checkpoint(cfg, checkpoint):
    model_cfg = checkpoint.get("gnn_handcrafted_model_config", {})
    cfg.gnn_handcrafted.hidden_channels = int(
        model_cfg.get("hidden_channels", cfg.gnn_handcrafted.get("hidden_channels", 32))
    )
    cfg.gnn_handcrafted.out_channels = int(
        model_cfg.get("out_channels", cfg.gnn_handcrafted.get("out_channels", 16))
    )
    cfg.gnn_handcrafted.decoder_hidden = int(
        model_cfg.get("decoder_hidden", cfg.gnn_handcrafted.get("decoder_hidden", 32))
    )
    cfg.gnn_handcrafted.dropout = float(
        model_cfg.get("dropout", cfg.gnn_handcrafted.get("dropout", 0.5))
    )
    cfg.gnn_handcrafted.edge_dropout = float(
        model_cfg.get("edge_dropout", cfg.gnn_handcrafted.get("edge_dropout", 0.0))
    )
    cfg.gnn_handcrafted.feature_dropout = float(
        model_cfg.get("feature_dropout", cfg.gnn_handcrafted.get("feature_dropout", 0.0))
    )
    cfg.gnn_handcrafted.edge_feature_dropout = float(
        model_cfg.get(
            "edge_feature_dropout",
            cfg.gnn_handcrafted.get("edge_feature_dropout", 0.0),
        )
    )
    cfg.gnn_handcrafted.decoder_scale_init = float(
        model_cfg.get(
            "decoder_scale_init",
            cfg.gnn_handcrafted.get("decoder_scale_init", 3.0),
        )
    )
    cfg.gnn_handcrafted.edge_input_dim = int(
        model_cfg.get("edge_input_dim", cfg.gnn_handcrafted.get("edge_input_dim", 0))
    )
    cfg.gnn_handcrafted.node_feature_source = checkpoint.get(
        "node_feature_source",
        cfg.gnn_handcrafted.get("node_feature_source", "raw"),
    )
    cfg.gnn_handcrafted.node_input_mode = checkpoint.get(
        "node_input_mode",
        cfg.gnn_handcrafted.get("node_input_mode", "structural_only"),
    )


@torch.no_grad()
def predict_gnn(model, data, device):
    model.eval()
    data = data.to(device)
    logits = model(
        data.x,
        data.edge_index,
        data.edge_label_index,
        edge_attr=getattr(data, "edge_attr", None),
        edge_label_attr=getattr(data, "edge_label_attr", None),
    )
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return probs, preds


@torch.no_grad()
def predict_pair_mlp(model, features, device, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(features, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    all_probs = []
    for (xb,) in loader:
        logits = model(xb.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def build_pair_submit_features(cfg, topology_pairs):
    node_array, train_array, test_df = load_pair_arrays(cfg, include_test=True)
    node_features, remapping = build_node_feature_mapping(node_array, cfg)
    topo_graph = build_positive_graph(topology_pairs)
    feature_builder = cfg.pair_mlp.get("feature_builder", "pairwise_rich")
    node_pair_representation = cfg.pair_mlp.get("node_pair_representation", "concat")

    test_pairs = test_df.to_numpy().astype(np.int64)
    if feature_builder == "handcrafted":
        feature_names = resolve_handcrafted_feature_names(cfg)
        context = build_handcrafted_context(topo_graph, cfg)
        x_submit, valid_mask = build_handcrafted_feature_matrix_with_context(
            test_pairs,
            node_features,
            remapping,
            topo_graph,
            context,
            feature_names,
            fill_missing=True,
        )
    else:
        pagerank_map, katz_map = compute_centrality_maps(topo_graph, cfg)
        x_submit, valid_mask = build_pair_feature_matrix(
            test_pairs,
            node_features,
            remapping,
            topo_graph,
            pagerank_map,
            katz_map,
            fill_missing=True,
            node_pair_representation=node_pair_representation,
        )
    return x_submit, test_df, valid_mask


def fit_pair_mlp_full_train(cfg, device):
    node_array, train_array, test_df = load_pair_arrays(cfg, include_test=True)
    node_features, remapping = build_node_feature_mapping(node_array, cfg)
    feature_builder = cfg.pair_mlp.get("feature_builder", "pairwise_rich")
    node_pair_representation = cfg.pair_mlp.get("node_pair_representation", "concat")

    full_pairs = train_array[:, :2].astype(np.int64)
    full_labels = train_array[:, 2].astype(np.float32)
    topology_pairs = full_pairs[full_labels == 1]

    topo_graph = build_positive_graph(topology_pairs)
    if feature_builder == "handcrafted":
        feature_names = resolve_handcrafted_feature_names(cfg)
        context = build_handcrafted_context(topo_graph, cfg)
        x_full, valid_train = build_handcrafted_feature_matrix_with_context(
            full_pairs,
            node_features,
            remapping,
            topo_graph,
            context,
            feature_names,
            fill_missing=False,
        )
    else:
        pagerank_map, katz_map = compute_centrality_maps(topo_graph, cfg)
        x_full, valid_train = build_pair_feature_matrix(
            full_pairs,
            node_features,
            remapping,
            topo_graph,
            pagerank_map,
            katz_map,
            fill_missing=False,
            node_pair_representation=node_pair_representation,
        )
    if not valid_train.all():
        if feature_builder == "handcrafted":
            skipped = int((~valid_train).sum())
            print(f"Skipped rows with unseen node features in full-train fit: {skipped}")
            full_labels = full_labels[valid_train]
        else:
            raise ValueError("Missing node features detected in pair MLP full-train pipeline.")

    if feature_builder == "handcrafted":
        x_submit, valid_mask = build_handcrafted_feature_matrix_with_context(
            test_df.to_numpy().astype(np.int64),
            node_features,
            remapping,
            topo_graph,
            context,
            feature_names,
            fill_missing=True,
        )
    else:
        x_submit, valid_mask = build_pair_feature_matrix(
            test_df.to_numpy().astype(np.int64),
            node_features,
            remapping,
            topo_graph,
            pagerank_map,
            katz_map,
            fill_missing=True,
            node_pair_representation=node_pair_representation,
        )

    x_full_s, scaler_stats = fit_feature_scaler(x_full)
    x_submit_s = apply_feature_scaler(x_submit, scaler_stats)

    model = build_model(cfg, x_full_s.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pair_mlp.get(
            "handcrafted_lr" if feature_builder == "handcrafted" else "lr",
            cfg.training.lr,
        ),
        weight_decay=cfg.pair_mlp.get(
            "handcrafted_wd" if feature_builder == "handcrafted" else "wd",
            cfg.training.wd,
        ),
    )

    batch_size = cfg.pair_mlp.get("batch_size", 256)
    full_train_epochs = cfg.inference.get(
        "full_train_epochs",
        cfg.pair_mlp.get("full_train_epochs", 6),
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_full_s, dtype=torch.float32),
            torch.tensor(full_labels, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(1, full_train_epochs + 1):
        model.train()
        running_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_local = xb.size(0)
            running_loss += loss.item() * batch_size_local
            total_count += batch_size_local

        print(
            f"Epoch {epoch:02d}/{full_train_epochs} | "
            f"loss={running_loss / max(total_count, 1):.6f}"
        )

    probs, preds = predict_pair_mlp(model, x_submit_s, device, batch_size=batch_size)
    return probs, preds, test_df, valid_mask


def predict_pair_mlp_from_checkpoint(cfg, device):
    checkpoint = torch.load(cfg.inference.checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "pair_mlp":
        raise ValueError("Expected a pair_mlp checkpoint dictionary.")

    topology_pairs = np.asarray(checkpoint["topology_train_pairs"].cpu().numpy(), dtype=np.int64)
    feature_builder = checkpoint.get("feature_builder", cfg.pair_mlp.get("feature_builder", "pairwise_rich"))
    node_feature_source = checkpoint.get("node_feature_source", cfg.pair_mlp.get("node_feature_source", "raw"))
    node_csv_has_header = checkpoint.get("node_csv_has_header", cfg.pair_mlp.get("node_csv_has_header", "auto"))
    node_pair_representation = checkpoint.get(
        "node_pair_representation",
        cfg.pair_mlp.get("node_pair_representation", "concat"),
    )
    handcrafted_feature_names = checkpoint.get("handcrafted_feature_names", None)
    cfg.pair_mlp.feature_builder = feature_builder
    restore_pair_mlp_cfg_from_checkpoint(cfg, checkpoint)
    node_array, _, test_df = load_pair_arrays(
        cfg,
        include_test=True,
        node_csv_has_header=node_csv_has_header if isinstance(node_csv_has_header, bool) else None,
    )
    node_features, remapping = build_node_feature_mapping(
        node_array,
        cfg,
        feature_source=node_feature_source,
    )
    topo_graph = build_positive_graph(topology_pairs)
    if feature_builder == "handcrafted":
        context = build_handcrafted_context(topo_graph, cfg)
        feature_names = resolve_handcrafted_feature_names(
            cfg,
            explicit_feature_names=handcrafted_feature_names,
        )
        x_submit, valid_mask = build_handcrafted_feature_matrix_with_context(
            test_df.to_numpy().astype(np.int64),
            node_features,
            remapping,
            topo_graph,
            context,
            feature_names,
            fill_missing=True,
        )
    else:
        pagerank_map, katz_map = compute_centrality_maps(topo_graph, cfg)
        x_submit, valid_mask = build_pair_feature_matrix(
            test_df.to_numpy().astype(np.int64),
            node_features,
            remapping,
            topo_graph,
            pagerank_map,
            katz_map,
            fill_missing=True,
            node_pair_representation=node_pair_representation,
        )
    x_submit_s = apply_feature_scaler(
        x_submit,
        {
            "mean": checkpoint["scaler_mean"].cpu().numpy(),
            "scale": checkpoint["scaler_scale"].cpu().numpy(),
        },
    )

    model = build_model(cfg, x_submit_s.shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    probs, preds = predict_pair_mlp(
        model,
        x_submit_s,
        device,
        batch_size=cfg.pair_mlp.get("batch_size", 256),
    )
    return probs, preds, test_df, valid_mask


def save_submission(test_df, probs, preds, valid_mask):
    print(f"Valid test edges: {valid_mask.sum()} / {len(valid_mask)}")
    print(f"Unknown-node test edges: {(~valid_mask).sum()}")

    assert len(preds) == test_df.height, f"Nombre de prédictions inattendu: {len(preds)}"
    assert test_df.height == 3498, f"Nombre de lignes test inattendu: {test_df.height}"

    submission = pl.DataFrame({"ID": np.arange(len(preds)), "Predicted": preds})
    submission.write_csv("submission.csv")
    print("submission.csv écrit avec", submission.height, "lignes")
    print(submission.head())

    submission_proba = pl.DataFrame({"ID": np.arange(len(probs)), "Predicted": probs})
    submission_proba.write_csv("submission_proba.csv")
    print("submission_proba.csv écrit avec", submission_proba.height, "lignes")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_kind = cfg.model.get("kind", "gnn")

    if model_kind == "pair_mlp":
        if cfg.inference.get("refit_on_full_train", False):
            probs, preds, test_df, valid_mask = fit_pair_mlp_full_train(cfg, device)
        else:
            probs, preds, test_df, valid_mask = predict_pair_mlp_from_checkpoint(cfg, device)
        save_submission(test_df, probs, preds, valid_mask)
    elif model_kind == "gnn_handcrafted":
        checkpoint = torch.load(cfg.inference.checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "gnn_handcrafted":
            raise ValueError("Expected a gnn_handcrafted checkpoint dictionary.")
        restore_handcrafted_gnn_cfg_from_checkpoint(cfg, checkpoint)
        test_data, test_df, valid_mask = build_handcrafted_gnn_test_data(cfg, checkpoint)
        model = build_model(cfg, test_data.x.size(1)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        probs, preds = predict_gnn(model, test_data, device)
        save_submission(test_df, probs, preds, valid_mask)
    else:
        test_data, test_df, valid_mask = build_gnn_test_data(cfg)
        model = build_model(cfg, test_data.x.size(1)).to(device)
        state = torch.load(cfg.inference.checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        probs, preds = predict_gnn(model, test_data, device)
        save_submission(test_df, probs, preds, valid_mask)

    if cfg.inference.submit:
        cmd = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            "centralesupelec-mlns-2026",
            "-f",
            "submission_proba.csv",
            "-m",
            cfg.inference.message,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
