import torch 
from torch_geometric.data import Data
import polars 
import numpy as np

def make_datasets(cfg) : 
    data_node = polars.read_csv("../../../" + cfg.data.DATA_BASE_PATH + "node_information.csv")
    edges_df = polars.read_csv("../../../" + cfg.data.DATA_BASE_PATH + "train.txt", separator=" ", has_header=False, new_columns=["a", "b", "label"])
    node_array = data_node.to_numpy()
    edge_array = edges_df.to_numpy()

    # create the feature dataset 
    # re-index the features 
    X_gcn = []
    remapping = {}
    # create the features 
    for i, row in enumerate(node_array) : 
        node_id = row[0]
        features = np.array(row[1:],dtype=np.float32)
        remapping[node_id] = i 
        X_gcn.append(features)
    X_gcn = torch.tensor(np.array(X_gcn), dtype=torch.float) # features 

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
