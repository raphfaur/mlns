import math

import networkx as nx
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from make_dataset import build_node_features, resolve_data_path


def resolve_node_csv_has_header(cfg):
    pair_cfg = cfg.get("pair_mlp", {})
    value = pair_cfg.get("node_csv_has_header", "auto")
    if isinstance(value, bool):
        return value
    if value != "auto":
        raise ValueError(f"Unknown pair_mlp.node_csv_has_header value: {value}")

    feature_builder = pair_cfg.get("feature_builder", "pairwise_rich")
    notebook_compat = pair_cfg.get("notebook_compat", False)
    return bool(feature_builder == "handcrafted" and notebook_compat)


def load_pair_arrays(cfg, include_test=False, node_csv_has_header=None):
    node_array = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "node_information.csv"),
        has_header=resolve_node_csv_has_header(cfg) if node_csv_has_header is None else node_csv_has_header,
    ).to_numpy()
    train_array = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "train.txt"),
        separator=" ",
        has_header=False,
        new_columns=["a", "b", "label"],
    ).to_numpy()

    if not include_test:
        return node_array, train_array

    test_df = pl.read_csv(
        resolve_data_path(cfg.data.DATA_BASE_PATH, "test.txt"),
        separator=" ",
        has_header=False,
        new_columns=["a", "b"],
    )
    return node_array, train_array, test_df


def build_raw_node_feature_mapping(node_array):
    remapping = {}
    features = []

    for i, row in enumerate(node_array):
        node_id = int(row[0])
        remapping[node_id] = i
        features.append(np.array(row[1:], dtype=np.float32))

    return np.asarray(features, dtype=np.float32), remapping


def build_node_feature_mapping(node_array, cfg, feature_source=None):
    pair_cfg = cfg.get("pair_mlp", {})
    feature_source = feature_source or pair_cfg.get("node_feature_source", "raw")

    if feature_source == "raw":
        return build_raw_node_feature_mapping(node_array)

    if feature_source == "preprocessed":
        node_features, remapping = build_node_features(node_array, cfg)
        return node_features.cpu().numpy().astype(np.float32), remapping

    raise ValueError(f"Unknown pair_mlp.node_feature_source: {feature_source}")


def build_positive_graph(positive_pairs):
    graph = nx.Graph()
    graph.add_edges_from((int(a), int(b)) for a, b in positive_pairs)
    return graph


def compute_centrality_maps(graph, cfg):
    pair_cfg = cfg.get("pair_mlp", {})
    pagerank_map = nx.pagerank(graph, alpha=pair_cfg.get("pagerank_alpha", 0.85))
    katz_map = nx.katz_centrality(
        graph,
        alpha=pair_cfg.get("katz_alpha", 0.005),
        beta=pair_cfg.get("katz_beta", 1.0),
        max_iter=pair_cfg.get("katz_max_iter", 1000),
        tol=pair_cfg.get("katz_tol", 1e-6),
    )
    return pagerank_map, katz_map


def topo_pair_features(graph, a, b):
    if a not in graph or b not in graph:
        return np.zeros(8, dtype=np.float32)

    common = list(nx.common_neighbors(graph, a, b))
    cn = float(len(common))
    aa = 0.0
    ra = 0.0
    for z in common:
        deg = graph.degree(z)
        if deg > 1:
            aa += 1.0 / math.log(deg)
        if deg > 0:
            ra += 1.0 / deg

    neighbors_a = set(graph.neighbors(a))
    neighbors_b = set(graph.neighbors(b))
    inter = len(neighbors_a & neighbors_b)
    union = len(neighbors_a | neighbors_b)
    jac = float(inter / union) if union > 0 else 0.0
    pa = float(len(neighbors_a) * len(neighbors_b))

    deg_a = float(len(neighbors_a))
    deg_b = float(len(neighbors_b))
    deg_diff = abs(deg_a - deg_b)

    return np.array(
        [
            cn,
            np.log1p(aa),
            np.log1p(ra),
            jac,
            np.log1p(pa),
            np.log1p(deg_a),
            np.log1p(deg_b),
            np.log1p(deg_diff),
        ],
        dtype=np.float32,
    )


def centrality_pair_features(a, b, pagerank_map, katz_map):
    pr_a = float(pagerank_map.get(a, 0.0))
    pr_b = float(pagerank_map.get(b, 0.0))
    kz_a = float(katz_map.get(a, 0.0))
    kz_b = float(katz_map.get(b, 0.0))
    return np.array(
        [
            np.log1p(pr_a),
            np.log1p(pr_b),
            np.log1p(abs(pr_a - pr_b)),
            np.log1p(kz_a),
            np.log1p(kz_b),
            np.log1p(abs(kz_a - kz_b)),
        ],
        dtype=np.float32,
    )


def cosine_similarity_feature(fa, fb):
    return float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-8))


def build_node_pair_representation(fa, fb, representation="concat"):
    if representation == "concat":
        return np.concatenate([fa, fb]).astype(np.float32)

    if representation == "cosine":
        return np.array([cosine_similarity_feature(fa, fb)], dtype=np.float32)

    if representation in {"cosine_no_duplicate", "cosine_no_duplictae"}:
        return np.array([cosine_similarity_feature(fa, fb)], dtype=np.float32)

    raise ValueError(f"Unknown pair_mlp.node_pair_representation: {representation}")


def extra_pair_features(fa, fb, include_cosine=True):
    diff = fa - fb
    abs_diff = np.abs(diff)
    prod = fa * fb

    l1 = float(abs_diff.sum())
    l2 = float(np.sqrt((diff ** 2).sum()))
    cos = cosine_similarity_feature(fa, fb)
    dot = float(np.dot(fa, fb))

    mean_abs = float(abs_diff.mean())
    max_abs = float(abs_diff.max())
    mean_prod = float(prod.mean())
    std_prod = float(prod.std())

    features = [l1, l2, dot, mean_abs, max_abs, mean_prod, std_prod]
    if include_cosine:
        features.insert(2, cos)

    return np.array(features, dtype=np.float32)


def build_pair_feature_matrix(
    pair_array,
    node_features,
    remapping,
    graph,
    pagerank_map,
    katz_map,
    fill_missing=False,
    node_pair_representation="concat",
):
    features = []
    valid_mask = []
    zero_node = np.zeros(node_features.shape[1], dtype=np.float32)

    for row in pair_array:
        a = int(row[0])
        b = int(row[1])
        valid = a in remapping and b in remapping
        valid_mask.append(valid)

        if valid:
            fa = node_features[remapping[a]]
            fb = node_features[remapping[b]]
        elif fill_missing:
            fa = zero_node
            fb = zero_node
        else:
            continue

        node_pair = build_node_pair_representation(
            fa,
            fb,
            representation=node_pair_representation,
        )
        topo = topo_pair_features(graph, a, b)
        include_cosine = node_pair_representation not in {
            "cosine_no_duplicate",
            "cosine_no_duplictae",
        }
        extra = extra_pair_features(fa, fb, include_cosine=include_cosine)
        cent = centrality_pair_features(a, b, pagerank_map, katz_map)
        features.append(np.concatenate([node_pair, topo, extra, cent]).astype(np.float32))

    return np.array(features, dtype=np.float32), np.array(valid_mask, dtype=bool)


def fit_feature_scaler(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features).astype(np.float32)
    stats = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32),
    }
    return scaled, stats


def apply_feature_scaler(features, scaler_stats):
    mean = np.asarray(scaler_stats["mean"], dtype=np.float32)
    scale = np.asarray(scaler_stats["scale"], dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    return ((features - mean) / scale).astype(np.float32)
