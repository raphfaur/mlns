import yaml
from sklearn.model_selection import train_test_split
import torch 
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
import hydra 
from omegaconf import DictConfig, OmegaConf

from eval import evaluate
from models import LinkPredictor
from make_dataset import make_datasets



    
    

def train_step(model,optimizer,train_data,criterion):
    model.train()
    optimizer.zero_grad()

    logits = model(
        train_data.x,
        train_data.edge_index,
        train_data.edge_label_index
    )

    loss = criterion(logits, train_data.edge_label.float())
    loss.backward()
    optimizer.step()
    return loss.item()

@hydra.main(version_base=None,config_path="conf",config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # load the data 
    graph_data = make_datasets(cfg)
    # split the train/test/val
    edge_index = graph_data.edge_index
    edge_label = graph_data.edge_label
    X_gcn = graph_data.x
    # edge_index: [2, E]
    # edge_label: [E]
    # only the positive edges here 
    pos_mask = edge_label == 1
    message_edge_index = edge_index[:, pos_mask]
    # split over the supervised samples 
    all_idx = torch.arange(edge_index.size(1))
    train_idx, temp_idx = train_test_split(
        all_idx.numpy(), test_size=cfg.data.test_size, random_state=42, stratify=edge_label.numpy()
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=edge_label.numpy()[temp_idx]
    )

    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_idx)

    train_data = Data(
        x=X_gcn,
        edge_index=message_edge_index,
        edge_label_index=edge_index[:, train_idx],
        edge_label=edge_label[train_idx]
    )

    val_data = Data(
        x=X_gcn,
        edge_index=message_edge_index,
        edge_label_index=edge_index[:, val_idx],
        edge_label=edge_label[val_idx]
    )

    test_data = Data(
        x=X_gcn,
        edge_index=message_edge_index,
        edge_label_index=edge_index[:, test_idx],
        edge_label=edge_label[test_idx]
    )

    # 

    # device
    device = torch.device("mps")

    # instantiate the model
    in_channels = train_data.x.size(1)
    model = LinkPredictor(in_channels, cfg.model.hidden_channels, cfg.model.out_channels).to(device)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    optimizer = AdamW(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()



    for epoch in range(1, cfg.training.epochs):
        loss = train_step(model,optimizer,train_data,criterion)
        if epoch % 100 == 0:
            val_acc, val_auc, val_ap = evaluate(val_data,model)
            print(f"epoch {epoch:03d}, loss: {loss:.4f}, val acc {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")

    test_acc, test_auc, test_ap = evaluate(test_data,model)
    print(f"test acc {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")

if __name__ == "__main__" :
    main()
    
    
    