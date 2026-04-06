import subprocess
import numpy as np
import polars as pl
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

from models import LinkPredictor


def build_test_data(cfg):
    base = cfg.data.DATA_BASE_PATH
    data_node = pl.read_csv("../../../" + base + "node_information.csv")
    train_df = pl.read_csv(
        "../../../" + base + "train.txt",
        separator=" ",
        has_header=False,
        new_columns=["a", "b", "label"],
    )
    test_df = pl.read_csv(
        "../../../" + base + "test.txt",
        separator=" ",
        has_header=False,
        new_columns=["a", "b"],
    )
    node_array = data_node.to_numpy()
    train_array = train_df.to_numpy()
    test_array = test_df.to_numpy()

    
    remapping = {}
    X_gcn = []

    for i, row in enumerate(node_array):
        node_id = row[0]
        features = np.array(row[1:], dtype=np.float32)
        remapping[node_id] = i
        X_gcn.append(features)

    X_gcn = torch.tensor(np.array(X_gcn), dtype=torch.float)

    
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

        i = remapping[node_i]
        j = remapping[node_j]

        sender.append(i)
        receiver.append(j)

    edge_index = torch.tensor([sender, receiver], dtype=torch.long)

    
    
    test_sender = []
    test_receiver = []
    valid_mask = []
    for edge in test_array:
        node_i, node_j = edge[0], edge[1]

        if node_i in remapping and node_j in remapping:
            i = remapping[node_i]
            j = remapping[node_j]
            test_sender.append(i)
            test_receiver.append(j)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    test_edge_label_index = torch.tensor(
        [test_sender, test_receiver], dtype=torch.long
    )

    test_data = Data(
        x=X_gcn,
        edge_index=edge_index,
        edge_label_index=test_edge_label_index,
    )

    return test_data, test_df, np.array(valid_mask) 

    


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    data = data.to(device)

    logits = model(data.x, data.edge_index, data.edge_label_index)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    return probs, preds


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    test_data, test_df, valid_mask = build_test_data(cfg)

    in_channels = test_data.x.size(1)
    model = LinkPredictor(
        in_channels,
        cfg.model.hidden_channels,
        cfg.model.out_channels,
    ).to(device)

    state_dict = torch.load(cfg.inference.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    probs, preds = predict(model, test_data, device)

    full_preds = np.zeros(test_df.height, dtype=int)      # fallback = 0
    full_probs = np.zeros(test_df.height, dtype=float)    # fallback prob = 0.0

    full_preds[valid_mask] = preds
    full_probs[valid_mask] = probs

    print(f"Valid test edges: {valid_mask.sum()} / {len(valid_mask)}")
    print(f"Unknown-node test edges: {(~valid_mask).sum()}")

    # verification for kaggle 
    assert len(full_preds) == 3498, f"Nombre de prédictions inattendu: {len(preds)}"
    assert test_df.height == 3498, f"Nombre de lignes test inattendu: {test_df.height}"

    submission = pl.DataFrame({
    "ID": np.arange(len(full_preds)),
    "Predicted": full_preds,
    })

    submission.write_csv("submission.csv")
    print("submission.csv écrit avec", submission.height, "lignes")
    print(submission.head())

    if cfg.inference.submit:
        cmd = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            "centralesupelec-mlns-2026",
            "-f",
            "submission.csv",
            "-m",
            cfg.inference.message,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()