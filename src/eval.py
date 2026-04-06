from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import torch

@torch.no_grad()
def evaluate(split_data,model,criterion):
    model.eval()
    logits = model(
        split_data.x,
        split_data.edge_index,
        split_data.edge_label_index
    )
    loss = criterion(logits, split_data.edge_label.float())
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = split_data.edge_label.cpu().numpy()
    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs) # ? 
    y_pred = (probs >= 0.5).astype(int) 
    acc = accuracy_score(y_true,y_pred)
    return loss, acc, auc, ap
