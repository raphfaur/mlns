import math

import networkx as nx
import numpy as np


def resolve_graph_feature_cfg(cfg):
    if cfg is None:
        return {}

    model_cfg = cfg.get("model", {})
    model_kind = model_cfg.get("kind", None)
    if model_kind == "gnn_handcrafted":
        return cfg.get("gnn_handcrafted", {})
    return cfg.get("pair_mlp", {})


def cosine_similarity(fa, fb):
    denom = (np.linalg.norm(fa) * np.linalg.norm(fb)) + 1e-8
    return float(np.dot(fa, fb) / denom)


def common_neighbors(a, b, graph):
    if a not in graph or b not in graph:
        return []
    return list(nx.common_neighbors(graph, a, b))


def build_handcrafted_context(graph, cfg):
    graph_cfg = resolve_graph_feature_cfg(cfg)
    pagerank_map = nx.pagerank(graph, alpha=graph_cfg.get("pagerank_alpha", 0.85))
    katz_map = nx.katz_centrality(
        graph,
        alpha=graph_cfg.get("katz_alpha", 0.005),
        beta=graph_cfg.get("katz_beta", 1.0),
        max_iter=graph_cfg.get("katz_max_iter", 1000),
        tol=graph_cfg.get("katz_tol", 1e-6),
    )

    try:
        clustering_coefs = nx.clustering(graph)
    except Exception:
        clustering_coefs = {}

    triangles = {node: 0 for node in graph.nodes()}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1 :]:
                if graph.has_edge(n1, n2):
                    triangles[node] += 1

    return {
        "pagerank_map": pagerank_map,
        "katz_map": katz_map,
        "clustering_coefs": clustering_coefs,
        "triangles": triangles,
    }


def f_cosine_similarity(a, b, fa, fb, graph, context):
    return cosine_similarity(fa, fb)


def f_dot_product(a, b, fa, fb, graph, context):
    return float(np.dot(fa, fb))


def f_l2_distance(a, b, fa, fb, graph, context):
    return float(np.linalg.norm(fa - fb))


def f_common_neighbors(a, b, fa, fb, graph, context):
    return float(len(common_neighbors(a, b, graph)))


def f_adamic_adar(a, b, fa, fb, graph, context):
    score = 0.0
    for z in common_neighbors(a, b, graph):
        deg_z = graph.degree(z)
        if deg_z > 1:
            score += 1.0 / math.log(deg_z)
    return float(score)


def f_resource_allocation(a, b, fa, fb, graph, context):
    score = 0.0
    for z in common_neighbors(a, b, graph):
        deg_z = graph.degree(z)
        if deg_z > 0:
            score += 1.0 / deg_z
    return float(score)


def f_jaccard(a, b, fa, fb, graph, context):
    if a not in graph or b not in graph:
        return 0.0
    neighbors_a = set(graph.neighbors(a))
    neighbors_b = set(graph.neighbors(b))
    union = len(neighbors_a | neighbors_b)
    return float(len(neighbors_a & neighbors_b) / union) if union > 0 else 0.0


def f_preferential_attachment(a, b, fa, fb, graph, context):
    if a not in graph or b not in graph:
        return 0.0
    return float(graph.degree(a) * graph.degree(b))


def f_degree_a(a, b, fa, fb, graph, context):
    if a not in graph:
        return 0.0
    return float(graph.degree(a))


def f_degree_b(a, b, fa, fb, graph, context):
    if b not in graph:
        return 0.0
    return float(graph.degree(b))


def f_degree_diff(a, b, fa, fb, graph, context):
    if a not in graph or b not in graph:
        return 0.0
    return float(abs(graph.degree(a) - graph.degree(b)))


def f_adamic_adar_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_adamic_adar(a, b, fa, fb, graph, context)))


def f_resource_allocation_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_resource_allocation(a, b, fa, fb, graph, context)))


def f_preferential_attachment_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_preferential_attachment(a, b, fa, fb, graph, context)))


def f_degree_a_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_degree_a(a, b, fa, fb, graph, context)))


def f_degree_b_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_degree_b(a, b, fa, fb, graph, context)))


def f_degree_diff_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_degree_diff(a, b, fa, fb, graph, context)))


def f_pagerank_a_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(context["pagerank_map"].get(int(a), 0.0)))


def f_pagerank_b_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(context["pagerank_map"].get(int(b), 0.0)))


def f_pagerank_diff_log1p(a, b, fa, fb, graph, context):
    pa = float(context["pagerank_map"].get(int(a), 0.0))
    pb = float(context["pagerank_map"].get(int(b), 0.0))
    return float(np.log1p(abs(pa - pb)))


def f_katz_a_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(context["katz_map"].get(int(a), 0.0)))


def f_katz_b_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(context["katz_map"].get(int(b), 0.0)))


def f_katz_diff_log1p(a, b, fa, fb, graph, context):
    ka = float(context["katz_map"].get(int(a), 0.0))
    kb = float(context["katz_map"].get(int(b), 0.0))
    return float(np.log1p(abs(ka - kb)))


def f_clustering_coef_a(a, b, fa, fb, graph, context):
    return float(context["clustering_coefs"].get(int(a), 0.0))


def f_clustering_coef_b(a, b, fa, fb, graph, context):
    return float(context["clustering_coefs"].get(int(b), 0.0))


def f_triangle_count_a(a, b, fa, fb, graph, context):
    return float(context["triangles"].get(int(a), 0))


def f_triangle_count_b(a, b, fa, fb, graph, context):
    return float(context["triangles"].get(int(b), 0))


def f_salton_index(a, b, fa, fb, graph, context):
    deg_a = graph.degree(int(a)) if int(a) in graph else 0
    deg_b = graph.degree(int(b)) if int(b) in graph else 0
    if deg_a == 0 or deg_b == 0:
        return 0.0
    cn = float(len(common_neighbors(int(a), int(b), graph)))
    return cn / (np.sqrt(deg_a * deg_b) + 1e-8)


def f_sorensen_index(a, b, fa, fb, graph, context):
    deg_a = graph.degree(int(a)) if int(a) in graph else 0
    deg_b = graph.degree(int(b)) if int(b) in graph else 0
    if deg_a + deg_b == 0:
        return 0.0
    cn = float(len(common_neighbors(int(a), int(b), graph)))
    return (2.0 * cn) / (deg_a + deg_b)


def f_salton_index_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_salton_index(a, b, fa, fb, graph, context)))


def f_sorensen_index_log1p(a, b, fa, fb, graph, context):
    return float(np.log1p(f_sorensen_index(a, b, fa, fb, graph, context)))


FEATURE_REGISTRY = {
    "cosine_similarity": f_cosine_similarity,
    "dot_product": f_dot_product,
    "l2_distance": f_l2_distance,
    "common_neighbors": f_common_neighbors,
    "adamic_adar": f_adamic_adar,
    "resource_allocation": f_resource_allocation,
    "jaccard": f_jaccard,
    "preferential_attachment": f_preferential_attachment,
    "degree_a": f_degree_a,
    "degree_b": f_degree_b,
    "degree_diff": f_degree_diff,
    "adamic_adar_log1p": f_adamic_adar_log1p,
    "resource_allocation_log1p": f_resource_allocation_log1p,
    "preferential_attachment_log1p": f_preferential_attachment_log1p,
    "degree_a_log1p": f_degree_a_log1p,
    "degree_b_log1p": f_degree_b_log1p,
    "degree_diff_log1p": f_degree_diff_log1p,
    "pagerank_a_log1p": f_pagerank_a_log1p,
    "pagerank_b_log1p": f_pagerank_b_log1p,
    "pagerank_diff_log1p": f_pagerank_diff_log1p,
    "katz_a_log1p": f_katz_a_log1p,
    "katz_b_log1p": f_katz_b_log1p,
    "katz_diff_log1p": f_katz_diff_log1p,
    "clustering_coef_a": f_clustering_coef_a,
    "clustering_coef_b": f_clustering_coef_b,
    "triangle_count_a": f_triangle_count_a,
    "triangle_count_b": f_triangle_count_b,
    "salton_index": f_salton_index,
    "sorensen_index": f_sorensen_index,
    "salton_index_log1p": f_salton_index_log1p,
    "sorensen_index_log1p": f_sorensen_index_log1p,
}


FEATURE_PRESETS = {
    "best83": [
        "resource_allocation_log1p",
        "jaccard",
        "preferential_attachment_log1p",
        "degree_a_log1p",
        "degree_b_log1p",
        "degree_diff_log1p",
        "cosine_similarity",
        "katz_a_log1p",
        "katz_b_log1p",
        "katz_diff_log1p",
    ],
    "baseline_small": [
        "resource_allocation_log1p",
        "preferential_attachment_log1p",
        "degree_a_log1p",
        "degree_b_log1p",
        "cosine_similarity",
    ],
    "legacy_best_like": [
        "resource_allocation_log1p",
        "jaccard",
        "preferential_attachment_log1p",
        "degree_a_log1p",
        "degree_b_log1p",
        "degree_diff_log1p",
        "cosine_similarity",
        "pagerank_a_log1p",
        "pagerank_b_log1p",
        "pagerank_diff_log1p",
        "katz_a_log1p",
        "katz_b_log1p",
        "katz_diff_log1p",
        "clustering_coef_a",
        "clustering_coef_b",
        "triangle_count_a",
        "triangle_count_b",
        "salton_index",
        "sorensen_index",
        "salton_index_log1p",
        "sorensen_index_log1p",
    ],
    "all": list(FEATURE_REGISTRY.keys()),
}


def nf_degree(node_id, graph, context):
    return float(graph.degree(int(node_id))) if int(node_id) in graph else 0.0


def nf_degree_log1p(node_id, graph, context):
    return float(np.log1p(nf_degree(node_id, graph, context)))


def nf_pagerank(node_id, graph, context):
    return float(context["pagerank_map"].get(int(node_id), 0.0))


def nf_pagerank_log1p(node_id, graph, context):
    return float(np.log1p(nf_pagerank(node_id, graph, context)))


def nf_katz(node_id, graph, context):
    return float(context["katz_map"].get(int(node_id), 0.0))


def nf_katz_log1p(node_id, graph, context):
    return float(np.log1p(nf_katz(node_id, graph, context)))


def nf_clustering_coef(node_id, graph, context):
    return float(context["clustering_coefs"].get(int(node_id), 0.0))


def nf_triangle_count(node_id, graph, context):
    return float(context["triangles"].get(int(node_id), 0))


def nf_triangle_count_log1p(node_id, graph, context):
    return float(np.log1p(nf_triangle_count(node_id, graph, context)))


NODE_FEATURE_REGISTRY = {
    "degree": nf_degree,
    "degree_log1p": nf_degree_log1p,
    "pagerank": nf_pagerank,
    "pagerank_log1p": nf_pagerank_log1p,
    "katz": nf_katz,
    "katz_log1p": nf_katz_log1p,
    "clustering_coef": nf_clustering_coef,
    "triangle_count": nf_triangle_count,
    "triangle_count_log1p": nf_triangle_count_log1p,
}


NODE_FEATURE_PRESETS = {
    "structural_small": [
        "degree_log1p",
        "pagerank_log1p",
        "katz_log1p",
        "clustering_coef",
        "triangle_count_log1p",
    ],
    "centrality_only": [
        "degree_log1p",
        "pagerank_log1p",
        "katz_log1p",
    ],
    "all": list(NODE_FEATURE_REGISTRY.keys()),
}


def resolve_handcrafted_feature_names(cfg, explicit_feature_names=None):
    if explicit_feature_names is not None:
        feature_names = list(explicit_feature_names)
    else:
        pair_cfg = cfg.get("pair_mlp", {})
        configured = pair_cfg.get("handcrafted_selected_features", [])
        if configured:
            feature_names = list(configured)
        else:
            preset_name = pair_cfg.get("handcrafted_preset", "baseline_small")
            if preset_name not in FEATURE_PRESETS:
                raise ValueError(f"Unknown pair_mlp.handcrafted_preset: {preset_name}")
            feature_names = list(FEATURE_PRESETS[preset_name])

    unknown = [name for name in feature_names if name not in FEATURE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown handcrafted features: {unknown}")
    return feature_names


def resolve_node_handcrafted_feature_names(cfg, explicit_feature_names=None):
    if explicit_feature_names is not None:
        feature_names = list(explicit_feature_names)
    else:
        gnn_cfg = cfg.get("gnn_handcrafted", {})
        configured = gnn_cfg.get("node_selected_features", [])
        if configured:
            feature_names = list(configured)
        else:
            preset_name = gnn_cfg.get("node_preset", "structural_small")
            if preset_name not in NODE_FEATURE_PRESETS:
                raise ValueError(f"Unknown gnn_handcrafted.node_preset: {preset_name}")
            feature_names = list(NODE_FEATURE_PRESETS[preset_name])

    unknown = [name for name in feature_names if name not in NODE_FEATURE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown node handcrafted features: {unknown}")
    return feature_names


def build_handcrafted_feature_vector(a, b, fa, fb, graph, context, feature_names):
    values = [FEATURE_REGISTRY[name](a, b, fa, fb, graph, context) for name in feature_names]
    return np.asarray(values, dtype=np.float32)


def build_node_handcrafted_feature_vector(node_id, graph, context, feature_names):
    values = [NODE_FEATURE_REGISTRY[name](node_id, graph, context) for name in feature_names]
    return np.asarray(values, dtype=np.float32)


def build_handcrafted_feature_matrix_with_context(
    pair_array,
    node_features,
    remapping,
    graph,
    context,
    feature_names,
    fill_missing=False,
):
    features = []
    valid_mask = []
    zero_vec = np.zeros(len(feature_names), dtype=np.float32)

    for row in pair_array:
        a = int(row[0])
        b = int(row[1])
        valid = a in remapping and b in remapping
        valid_mask.append(valid)

        if valid:
            fa = node_features[remapping[a]]
            fb = node_features[remapping[b]]
            vec = build_handcrafted_feature_vector(a, b, fa, fb, graph, context, feature_names)
        elif fill_missing:
            vec = zero_vec
        else:
            continue

        features.append(vec)

    return np.asarray(features, dtype=np.float32), np.asarray(valid_mask, dtype=bool)


def build_node_handcrafted_feature_matrix(node_ids, graph, context, feature_names):
    features = [
        build_node_handcrafted_feature_vector(int(node_id), graph, context, feature_names)
        for node_id in node_ids
    ]
    return np.asarray(features, dtype=np.float32)
