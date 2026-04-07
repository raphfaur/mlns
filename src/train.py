import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from eval import evaluate as evaluate_gnn
from models import build_model
from make_dataset import make_datasets
from pair_features import (
    apply_feature_scaler,
    build_node_feature_mapping,
    build_pair_feature_matrix,
    build_positive_graph,
    compute_centrality_maps,
    fit_feature_scaler,
    load_pair_arrays,
)


def build_pair_training_cfg(cfg):
    pair_cfg = cfg.get("pair_mlp", {})
    base_cfg = cfg.training
    return {
        "epochs": pair_cfg.get("epochs", base_cfg.epochs),
        "lr": pair_cfg.get("lr", base_cfg.lr),
        "wd": pair_cfg.get("wd", base_cfg.wd),
        "eval_every": pair_cfg.get("eval_every", base_cfg.get("eval_every", 25)),
        "patience": pair_cfg.get("patience", base_cfg.get("patience", 40)),
        "min_delta": pair_cfg.get("min_delta", base_cfg.get("min_delta", 1e-4)),
        "label_smoothing": pair_cfg.get(
            "label_smoothing",
            base_cfg.get("label_smoothing", 0.0),
        ),
        "grad_clip": pair_cfg.get("grad_clip", base_cfg.get("grad_clip", None)),
        "use_scheduler": pair_cfg.get("use_scheduler", False),
        "lr_patience": pair_cfg.get("lr_patience", base_cfg.get("lr_patience", 5)),
        "lr_decay_factor": pair_cfg.get(
            "lr_decay_factor",
            base_cfg.get("lr_decay_factor", 0.5),
        ),
        "min_lr": pair_cfg.get("min_lr", base_cfg.get("min_lr", 1e-5)),
    }


def smooth_binary_targets(targets, label_smoothing):
    if label_smoothing <= 0.0:
        return targets
    return targets * (1.0 - label_smoothing) + 0.5 * label_smoothing


def train_gnn_step(model, optimizer, train_data, criterion, label_smoothing=0.0, grad_clip=None):
    model.train()
    optimizer.zero_grad()

    logits = model(
        train_data.x,
        train_data.edge_index,
        train_data.edge_label_index,
    )
    targets = smooth_binary_targets(train_data.edge_label.float(), label_smoothing)
    loss = criterion(logits, targets)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


def train_pair_mlp_step(model, optimizer, xb, yb, criterion, label_smoothing=0.0, grad_clip=None):
    model.train()
    optimizer.zero_grad()

    logits = model(xb)
    targets = smooth_binary_targets(yb.float(), label_smoothing)
    loss = criterion(logits, targets)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_pair_mlp(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb.float())

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(yb.cpu())

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)

    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, acc, auc, ap


def make_pair_loader(features, labels, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_with_early_stopping(
    train_cfg,
    model,
    optimizer,
    scheduler,
    criterion,
    train_epoch_fn,
    eval_fn,
    save_payload_fn,
):
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    grad_clip = train_cfg.get("grad_clip", None)
    best_val_auc = float("-inf")
    best_epoch = 0
    patience = train_cfg.get("patience", 40)
    min_delta = train_cfg.get("min_delta", 1e-4)
    patience_counter = 0
    eval_every = train_cfg.get("eval_every", 25)
    max_epochs = train_cfg.get("epochs", 100)

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch_fn(label_smoothing, grad_clip)
        wandb.log({"epoch": epoch, "train_loss": loss})

        if epoch % eval_every != 0 and epoch != max_epochs:
            continue

        val_loss, val_acc, val_auc, val_ap = eval_fn()
        print(
            f"epoch {epoch:03d}, loss: {loss:.4f}, val acc {val_acc:.4f}, "
            f"Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}"
        )
        wandb.log(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_ap": val_ap,
            }
        )

        if scheduler is not None:
            scheduler.step(val_auc)
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(save_payload_fn(), "best_model.pt")
            wandb.save("best_model.pt")
            print(f"!!! new best model saved (AUC={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch:03d} "
                    f"(best epoch {best_epoch:03d}, best val AUC {best_val_auc:.4f})"
                )
                break


def train_gnn(cfg, device):
    graph_data = make_datasets(cfg)
    edge_index = graph_data.edge_index
    edge_label = graph_data.edge_label
    x_gnn = graph_data.x
    print("Total number of nodes", x_gnn.shape[0])

    all_idx = torch.arange(edge_index.size(1))
    train_idx, temp_idx = train_test_split(
        all_idx.numpy(),
        test_size=cfg.data.test_size,
        random_state=42,
        stratify=edge_label.numpy(),
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=edge_label.numpy()[temp_idx],
    )

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    train_pos_mask = edge_label[train_idx] == 1
    train_pos_idx = train_idx[train_pos_mask]
    message_edge_index = edge_index[:, train_pos_idx]

    train_data = Data(
        x=x_gnn,
        edge_index=message_edge_index,
        edge_label_index=edge_index[:, train_idx],
        edge_label=edge_label[train_idx],
    )
    val_data = Data(
        x=x_gnn,
        edge_index=message_edge_index,
        edge_label_index=edge_index[:, val_idx],
        edge_label=edge_label[val_idx],
    )
    test_data = Data(
        x=x_gnn,
        edge_index=message_edge_index,
        edge_label_index=edge_index[:, test_idx],
        edge_label=edge_label[test_idx],
    )

    model = build_model(cfg, train_data.x.size(1)).to(device)
    wandb.watch(model, log="all")

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.training.get("lr_decay_factor", 0.5),
        patience=cfg.training.get("lr_patience", 5),
        min_lr=cfg.training.get("min_lr", 1e-5),
    )
    criterion = nn.BCEWithLogitsLoss()

    def train_epoch(label_smoothing, grad_clip):
        return train_gnn_step(
            model,
            optimizer,
            train_data,
            criterion,
            label_smoothing=label_smoothing,
            grad_clip=grad_clip,
        )

    def evaluate_val():
        return evaluate_gnn(val_data, model, criterion)

    def save_payload():
        return model.state_dict()

    train_with_early_stopping(
        cfg.training,
        model,
        optimizer,
        scheduler,
        criterion,
        train_epoch,
        evaluate_val,
        save_payload,
    )

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    _, test_acc, test_auc, test_ap = evaluate_gnn(test_data, model, criterion)
    wandb.log({"test_acc": test_acc, "test_auc": test_auc, "test_ap": test_ap})
    print(f"test acc {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")


def train_pair_mlp(cfg, device):
    node_array, train_array = load_pair_arrays(cfg, include_test=False)
    node_features, remapping = build_node_feature_mapping(node_array, cfg)
    pair_train_cfg = build_pair_training_cfg(cfg)

    pairs = train_array[:, :2].astype(np.int64)
    labels = train_array[:, 2].astype(np.float32)
    all_idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=cfg.data.test_size,
        random_state=42,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=labels[temp_idx],
    )

    train_pairs = pairs[train_idx]
    train_labels = labels[train_idx]
    val_pairs = pairs[val_idx]
    val_labels = labels[val_idx]
    test_pairs = pairs[test_idx]
    test_labels = labels[test_idx]

    topology_train_pairs = train_pairs[train_labels == 1]
    topo_graph = build_positive_graph(topology_train_pairs)
    pagerank_map, katz_map = compute_centrality_maps(topo_graph, cfg)

    x_train, train_valid = build_pair_feature_matrix(
        train_pairs,
        node_features,
        remapping,
        topo_graph,
        pagerank_map,
        katz_map,
        fill_missing=False,
    )
    x_val, val_valid = build_pair_feature_matrix(
        val_pairs,
        node_features,
        remapping,
        topo_graph,
        pagerank_map,
        katz_map,
        fill_missing=False,
    )
    x_test, test_valid = build_pair_feature_matrix(
        test_pairs,
        node_features,
        remapping,
        topo_graph,
        pagerank_map,
        katz_map,
        fill_missing=False,
    )

    if not train_valid.all() or not val_valid.all() or not test_valid.all():
        raise ValueError("Missing node features detected in pair MLP pipeline.")

    x_train_s, scaler_stats = fit_feature_scaler(x_train)
    x_val_s = apply_feature_scaler(x_val, scaler_stats)
    x_test_s = apply_feature_scaler(x_test, scaler_stats)

    batch_size = cfg.pair_mlp.get("batch_size", 256)
    train_loader = make_pair_loader(x_train_s, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = make_pair_loader(x_val_s, val_labels, batch_size=batch_size, shuffle=False)
    test_loader = make_pair_loader(x_test_s, test_labels, batch_size=batch_size, shuffle=False)

    model = build_model(cfg, x_train_s.shape[1]).to(device)
    wandb.watch(model, log="all")

    optimizer = AdamW(
        model.parameters(),
        lr=pair_train_cfg["lr"],
        weight_decay=pair_train_cfg["wd"],
    )
    scheduler = None
    if pair_train_cfg.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=pair_train_cfg.get("lr_decay_factor", 0.5),
            patience=pair_train_cfg.get("lr_patience", 5),
            min_lr=pair_train_cfg.get("min_lr", 1e-5),
        )
    criterion = nn.BCEWithLogitsLoss()

    def train_epoch(label_smoothing, grad_clip):
        running_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            batch_loss = train_pair_mlp_step(
                model,
                optimizer,
                xb,
                yb,
                criterion,
                label_smoothing=label_smoothing,
                grad_clip=grad_clip,
            )
            batch_size_local = xb.size(0)
            running_loss += batch_loss * batch_size_local
            total_count += batch_size_local
        return running_loss / max(total_count, 1)

    def evaluate_val():
        return evaluate_pair_mlp(val_loader, model, criterion, device)

    def save_payload():
        return {
            "kind": "pair_mlp",
            "model_state_dict": model.state_dict(),
            "scaler_mean": torch.tensor(scaler_stats["mean"], dtype=torch.float32),
            "scaler_scale": torch.tensor(scaler_stats["scale"], dtype=torch.float32),
            "topology_train_pairs": torch.tensor(topology_train_pairs, dtype=torch.long),
            "input_dim": int(x_train_s.shape[1]),
            "node_feature_source": cfg.pair_mlp.get("node_feature_source", "raw"),
        }

    train_with_early_stopping(
        pair_train_cfg,
        model,
        optimizer,
        scheduler,
        criterion,
        train_epoch,
        evaluate_val,
        save_payload,
    )

    checkpoint = torch.load("best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_acc, test_auc, test_ap = evaluate_pair_mlp(test_loader, model, criterion, device)
    wandb.log({"test_acc": test_acc, "test_auc": test_auc, "test_ap": test_ap})
    print(f"test acc {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")


def build_run_name(cfg):
    model_kind = cfg.model.get("kind", "gnn")
    if model_kind == "pair_mlp":
        pair_train_cfg = build_pair_training_cfg(cfg)
        return f"pairmlp_p{cfg.pair_mlp.get('proj_dim', 64)}_lr{pair_train_cfg['lr']}"
    return f"gnn_h{cfg.model.hidden_channels}_lr{cfg.training.lr}"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        project="link-prediction-gcn",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=build_run_name(cfg),
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_kind = cfg.model.get("kind", "gnn")

    if model_kind == "pair_mlp":
        train_pair_mlp(cfg, device)
        return

    if model_kind == "gnn":
        train_gnn(cfg, device)
        return

    raise ValueError(f"Unknown model.kind: {model_kind}")


if __name__ == "__main__":
    main()
