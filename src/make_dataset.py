import torch 
from torch_geometric.data import Data
import polars 
import numpy as np
from sklearn.decomposition import TruncatedSVD


def build_node_features(node_array, cfg):
    remapping = {}
    features = []

    for i, row in enumerate(node_array):
        node_id = row[0]
        remapping[node_id] = i
        features.append(np.array(row[1:], dtype=np.float32))

    X = np.array(features, dtype=np.float32)
    prep_cfg = cfg.get("preprocessing", {})

    if prep_cfg.get("log1p", False):
        X = np.log1p(X)

    if prep_cfg.get("use_truncated_svd", False):
        max_components = min(X.shape[0] - 1, X.shape[1] - 1)
        n_components = min(prep_cfg.get("svd_dim", 128), max_components)
        if n_components < 2:
            raise ValueError(
                f"Invalid TruncatedSVD dimension {n_components} for features of shape {X.shape}"
            )

        svd = TruncatedSVD(
            n_components=n_components,
            n_iter=prep_cfg.get("svd_n_iter", 7),
            random_state=prep_cfg.get("random_state", 42),
        )
        X = svd.fit_transform(X)
        explained_var = float(svd.explained_variance_ratio_.sum())
        print(
            f"TruncatedSVD reduced node features from {node_array.shape[1] - 1} "
            f"to {n_components} dims (explained variance {explained_var:.4f})"
        )

    X_gcn = torch.tensor(X.astype(np.float32), dtype=torch.float)
    return X_gcn, remapping

def make_datasets(cfg) : 
    data_node = polars.read_csv(
        "../../../" + cfg.data.DATA_BASE_PATH + "node_information.csv",
        has_header=False,
    )
    edges_df = polars.read_csv("../../../" + cfg.data.DATA_BASE_PATH + "train.txt", separator=" ", has_header=False, new_columns=["a", "b", "label"])
    node_array = data_node.to_numpy()
    edge_array = edges_df.to_numpy()

    # create the feature dataset 
    # re-index the features 
    X_gcn, remapping = build_node_features(node_array, cfg)

    # create the edges 
    edges = edge_array[:, :2]
    labels = edge_array[:, 2] 
    print("edges.shape:", edges.shape)    # attendu: (10496, 2)
    print("labels.shape:", labels.shape)  # attendu: (10496,)
    receiver = []
    sender = []
    edge_labels = []

    for edge, label in zip(edges,labels) :
        
        node_i, node_j = edge[0], edge[1]
        if node_i not in remapping or node_j not in remapping:
            continue
        i = remapping[node_i]
        j = remapping[node_j]
        edge_labels.append(label)

        sender.append(i)
        receiver.append(j)

    edge_labels = np.array(edge_labels)
    print(edge_labels.shape)

    edge_index = torch.tensor([sender,receiver],dtype=torch.long)
    edge_label = torch.tensor(edge_labels, dtype=torch.float)

    graph_data = Data(
        x=X_gcn,
        edge_index=edge_index,
        edge_label_index=edge_index,
        edge_label=edge_label
    )
    return graph_data
