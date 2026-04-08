import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from handcrafted_features import (
    FEATURE_PRESETS,
    build_handcrafted_context,
    build_handcrafted_feature_matrix_with_context,
    build_node_handcrafted_feature_matrix,
    resolve_node_handcrafted_feature_names,
)
from make_dataset import build_node_features, resolve_data_path
from pair_features import build_positive_graph


def load_gnn_arrays(cfg, include_test=False):
    node_df = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "node_information.csv"),
        has_header=False,
    )
    train_df = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "train.txt"),
        separator=" ",
        has_header=False,
        new_columns=["a", "b", "label"],
    )
    test_df = None
    if include_test:
        test_df = pl.read_csv(
            resolve_data_path(cfg.data.DATA_BASE_PATH, "test.txt"),
            separator=" ",
            has_header=False,
            new_columns=["a", "b"],
        )
    return node_df.to_numpy(), train_df.to_numpy(), test_df


def build_gnn_base_node_mapping(node_array, cfg, feature_source=None):
    gnn_cfg = cfg.get("gnn_handcrafted", {})
    feature_source = feature_source or gnn_cfg.get("node_feature_source", "raw")

    if feature_source == "raw":
        remapping = {}
        features = []
        for idx, row in enumerate(node_array):
            node_id = int(row[0])
            remapping[node_id] = idx
            features.append(np.asarray(row[1:], dtype=np.float32))
        return np.asarray(features, dtype=np.float32), remapping

    if feature_source == "preprocessed":
        node_features, remapping = build_node_features(node_array, cfg)
        return node_features.cpu().numpy().astype(np.float32), remapping

    raise ValueError(f"Unknown gnn_handcrafted.node_feature_source: {feature_source}")


def fit_feature_stats(features):
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean, "scale": scale}


def apply_feature_stats(features, stats):
    return ((features - stats["mean"]) / stats["scale"]).astype(np.float32)


def resolve_gnn_edge_feature_names(cfg, explicit_feature_names=None):
    if explicit_feature_names is not None:
        feature_names = list(explicit_feature_names)
    else:
        gnn_cfg = cfg.get("gnn_handcrafted", {})
        configured = gnn_cfg.get("edge_selected_features", [])
        if configured:
            feature_names = list(configured)
        else:
            preset_name = gnn_cfg.get("edge_preset", "best83")
            if preset_name not in FEATURE_PRESETS:
                raise ValueError(f"Unknown gnn_handcrafted.edge_preset: {preset_name}")
            feature_names = list(FEATURE_PRESETS[preset_name])
    unknown = [name for name in feature_names if name not in FEATURE_PRESETS["all"]]
    if unknown:
        raise ValueError(f"Unknown edge handcrafted features: {unknown}")
    return feature_names


def build_gnn_node_inputs(cfg, node_array, topo_graph, context, node_feature_names, base_node_features):
    node_ids = node_array[:, 0].astype(np.int64)
    node_structural = build_node_handcrafted_feature_matrix(
        node_ids,
        topo_graph,
        context,
        node_feature_names,
    )
    node_input_mode = cfg.gnn_handcrafted.get("node_input_mode", "structural_only")
    if node_input_mode == "structural_only":
        node_inputs = node_structural
    elif node_input_mode == "base_plus_structural":
        node_inputs = np.concatenate([base_node_features, node_structural], axis=1)
    elif node_input_mode == "base_only":
        node_inputs = base_node_features
    else:
        raise ValueError(f"Unknown gnn_handcrafted.node_input_mode: {node_input_mode}")
    return node_inputs.astype(np.float32)


def _mapped_pairs_and_labels(node_array, train_array, cfg):
    base_node_features, remapping = build_gnn_base_node_mapping(node_array, cfg)
    pairs = train_array[:, :2].astype(np.int64)
    labels = train_array[:, 2].astype(np.float32)

    valid_mask = np.array(
        [(int(a) in remapping) and (int(b) in remapping) for a, b in pairs],
        dtype=bool,
    )
    if not valid_mask.all():
        pairs = pairs[valid_mask]
        labels = labels[valid_mask]

    mapped_edge_index = np.asarray(
        [[remapping[int(a)], remapping[int(b)]] for a, b in pairs],
        dtype=np.int64,
    ).T
    return base_node_features, remapping, pairs, labels, mapped_edge_index


def build_handcrafted_gnn_train_datasets(cfg):
    node_array, train_array, _ = load_gnn_arrays(cfg, include_test=False)
    base_node_features, remapping, pairs, labels, mapped_edge_index = _mapped_pairs_and_labels(
        node_array,
        train_array,
        cfg,
    )

    all_idx = np.arange(pairs.shape[0])
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

    topology_pairs = pairs[train_idx][labels[train_idx] == 1]
    topo_graph = build_positive_graph(topology_pairs)
    context = build_handcrafted_context(topo_graph, cfg)

    node_feature_names = resolve_node_handcrafted_feature_names(cfg)
    edge_feature_names = resolve_gnn_edge_feature_names(cfg)

    node_inputs = build_gnn_node_inputs(
        cfg,
        node_array,
        topo_graph,
        context,
        node_feature_names,
        base_node_features,
    )
    node_stats = fit_feature_stats(node_inputs)
    node_inputs_s = apply_feature_stats(node_inputs, node_stats)

    edge_features_all, edge_valid_mask = build_handcrafted_feature_matrix_with_context(
        pairs,
        base_node_features,
        remapping,
        topo_graph,
        context,
        edge_feature_names,
        fill_missing=False,
    )
    if not edge_valid_mask.all():
        raise ValueError("Missing node features detected in handcrafted GNN edge pipeline.")

    edge_stats = fit_feature_stats(edge_features_all[train_idx])
    edge_features_all_s = apply_feature_stats(edge_features_all, edge_stats)

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    mapped_edge_index_t = torch.tensor(mapped_edge_index, dtype=torch.long)
    node_inputs_t = torch.tensor(node_inputs_s, dtype=torch.float32)
    edge_features_all_t = torch.tensor(edge_features_all_s, dtype=torch.float32)

    train_pos_mask = labels_t[train_idx_t] == 1
    train_pos_idx_t = train_idx_t[train_pos_mask]
    message_edge_index = mapped_edge_index_t[:, train_pos_idx_t]
    message_edge_attr = edge_features_all_t[train_pos_idx_t]

    def make_split(split_idx_t):
        return Data(
            x=node_inputs_t,
            edge_index=message_edge_index,
            edge_attr=message_edge_attr,
            edge_label_index=mapped_edge_index_t[:, split_idx_t],
            edge_label=labels_t[split_idx_t],
            edge_label_attr=edge_features_all_t[split_idx_t],
        )

    return {
        "train_data": make_split(train_idx_t),
        "val_data": make_split(val_idx_t),
        "test_data": make_split(test_idx_t),
        "topology_pairs": topology_pairs,
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
        "node_stats": node_stats,
        "edge_stats": edge_stats,
        "node_feature_source": cfg.gnn_handcrafted.get("node_feature_source", "raw"),
        "node_input_mode": cfg.gnn_handcrafted.get("node_input_mode", "structural_only"),
        "node_input_dim": int(node_inputs_s.shape[1]),
        "edge_input_dim": int(edge_features_all_s.shape[1]),
    }


def build_handcrafted_gnn_test_data(cfg, checkpoint):
    node_array, train_array, test_df = load_gnn_arrays(cfg, include_test=True)
    node_feature_source = checkpoint.get(
        "node_feature_source",
        cfg.gnn_handcrafted.get("node_feature_source", "raw"),
    )
    base_node_features, remapping = build_gnn_base_node_mapping(
        node_array,
        cfg,
        feature_source=node_feature_source,
    )

    topology_pairs = np.asarray(checkpoint["topology_pairs"], dtype=np.int64)
    topo_graph = build_positive_graph(topology_pairs)
    context = build_handcrafted_context(topo_graph, cfg)

    node_feature_names = checkpoint.get("node_feature_names", resolve_node_handcrafted_feature_names(cfg))
    edge_feature_names = checkpoint.get("edge_feature_names", resolve_gnn_edge_feature_names(cfg))

    node_inputs = build_gnn_node_inputs(
        cfg,
        node_array,
        topo_graph,
        context,
        node_feature_names,
        base_node_features,
    )
    node_inputs_s = apply_feature_stats(
        node_inputs,
        {
            "mean": checkpoint["node_scaler_mean"].cpu().numpy(),
            "scale": checkpoint["node_scaler_scale"].cpu().numpy(),
        },
    )

    message_edge_index = np.asarray(
        [[remapping[int(a)], remapping[int(b)]] for a, b in topology_pairs],
        dtype=np.int64,
    ).T
    message_edge_features, message_valid_mask = build_handcrafted_feature_matrix_with_context(
        topology_pairs,
        base_node_features,
        remapping,
        topo_graph,
        context,
        edge_feature_names,
        fill_missing=False,
    )
    if not message_valid_mask.all():
        raise ValueError("Missing node features detected in handcrafted GNN message edges.")
    message_edge_features_s = apply_feature_stats(
        message_edge_features,
        {
            "mean": checkpoint["edge_scaler_mean"].cpu().numpy(),
            "scale": checkpoint["edge_scaler_scale"].cpu().numpy(),
        },
    )

    test_pairs = test_df.to_numpy().astype(np.int64)
    valid_mask = np.array(
        [(int(a) in remapping) and (int(b) in remapping) for a, b in test_pairs],
        dtype=bool,
    )
    valid_test_pairs = test_pairs[valid_mask]
    test_edge_index = np.asarray(
        [[remapping[int(a)], remapping[int(b)]] for a, b in valid_test_pairs],
        dtype=np.int64,
    ).T
    test_edge_features, test_valid_mask = build_handcrafted_feature_matrix_with_context(
        valid_test_pairs,
        base_node_features,
        remapping,
        topo_graph,
        context,
        edge_feature_names,
        fill_missing=False,
    )
    if not test_valid_mask.all():
        raise ValueError("Missing node features detected in handcrafted GNN test edges.")
    test_edge_features_s = apply_feature_stats(
        test_edge_features,
        {
            "mean": checkpoint["edge_scaler_mean"].cpu().numpy(),
            "scale": checkpoint["edge_scaler_scale"].cpu().numpy(),
        },
    )

    data = Data(
        x=torch.tensor(node_inputs_s, dtype=torch.float32),
        edge_index=torch.tensor(message_edge_index, dtype=torch.long),
        edge_attr=torch.tensor(message_edge_features_s, dtype=torch.float32),
        edge_label_index=torch.tensor(test_edge_index, dtype=torch.long),
        edge_label_attr=torch.tensor(test_edge_features_s, dtype=torch.float32),
    )
    return data, test_df, valid_mask
