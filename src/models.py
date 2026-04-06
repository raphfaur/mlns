import torch.nn.functional as F 
from torch import nn 
from torch_geometric.nn import GCNConv
import torch 

# === GNN Link predictor == 

# create the encoder 
class GCNencoder(nn.Module) : 
    def __init__(self,in_channels,hidden_channels,out_channels) :
        super().__init__()
        self.conv1 = GCNConv(in_channels,hidden_channels)
        self.conv2 = GCNConv(hidden_channels,out_channels)
    
    def forward(self,x,edge_index) :
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = self.conv2(x,edge_index)
        return x 

# create the decoder
class DotProductDecoder(nn.Module) : 
    def forward(self, z, edge_label_index) : 
        src, dst = edge_label_index
        return (z[src]*z[dst]).sum(dim=1)
    
# variant : MLP decoder, more powerful

class MLPdecoder(nn.Module) :
    def __init__(self,emb_dim,hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        h = torch.cat([z[src], z[dst]], dim=-1)
        return self.mlp(h).squeeze(-1)
 
# create the LinkPredictor : (u,v) -> {0,1} (presence or not)
# assembling the encoder and the decoder 
class LinkPredictor(nn.Module) :
    def __init__(self, in_channels, hidden_channels, out_channels) :
        super().__init__()
        self.encoder = GCNencoder(in_channels,hidden_channels,out_channels)
        self.decoder = DotProductDecoder()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        logits = self.decoder(z, edge_label_index)
        return logits

