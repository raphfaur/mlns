import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.3,
        edge_dropout=0.0,
        feature_dropout=0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.skip_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.feature_dropout = feature_dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.training and self.edge_dropout > 0.0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout)

        h = self.conv1(x, edge_index)
        h = self.norm1(h + x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.norm2(h + self.skip_proj(x))
        h = torch.tanh(h)
        return h


class ScaledCosineDecoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.3, scale_init=5.0, max_scale=30.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(scale_init))))
        self.max_scale = max_scale
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        z_src_raw = z[src]
        z_dst_raw = z[dst]
        pair_features = torch.cat(
            [z_src_raw, z_dst_raw, torch.abs(z_src_raw - z_dst_raw), z_src_raw * z_dst_raw],
            dim=-1,
        )
        mlp_score = self.edge_mlp(pair_features).squeeze(-1)

        z_src = F.normalize(z[src], p=2, dim=-1)
        z_dst = F.normalize(z[dst], p=2, dim=-1)
        cosine = (z_src * z_dst).sum(dim=-1)
        scale = torch.exp(self.log_scale).clamp(max=self.max_scale)
        return scale * cosine + mlp_score


class LinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.3,
        edge_dropout=0.0,
        feature_dropout=0.0,
        decoder_scale_init=5.0,
        decoder_hidden=None,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels,
            hidden_channels,
            out_channels,
            dropout=dropout,
            edge_dropout=edge_dropout,
            feature_dropout=feature_dropout,
        )
        self.decoder = ScaledCosineDecoder(
            emb_dim=out_channels,
            hidden_dim=decoder_hidden or hidden_channels,
            dropout=dropout,
            scale_init=decoder_scale_init,
        )

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        logits = self.decoder(z, edge_label_index)
        return logits


class PairFeatureMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        proj_dim=64,
        hidden_dim_1=64,
        hidden_dim_2=32,
        dropout_1=0.30,
        dropout_2=0.25,
        dropout_3=0.15,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(proj_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_3),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(cfg, in_dim):
    model_kind = cfg.model.get("kind", "gnn")

    if model_kind == "pair_mlp":
        pair_cfg = cfg.get("pair_mlp", {})
        return PairFeatureMLP(
            in_dim=in_dim,
            proj_dim=pair_cfg.get("proj_dim", 64),
            hidden_dim_1=pair_cfg.get("hidden_dim_1", 64),
            hidden_dim_2=pair_cfg.get("hidden_dim_2", 32),
            dropout_1=pair_cfg.get("dropout_1", 0.30),
            dropout_2=pair_cfg.get("dropout_2", 0.25),
            dropout_3=pair_cfg.get("dropout_3", 0.15),
        )

    if model_kind == "gnn":
        return LinkPredictor(
            in_channels=in_dim,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            dropout=cfg.model.get("dropout", 0.3),
            edge_dropout=cfg.model.get("edge_dropout", 0.0),
            feature_dropout=cfg.model.get("feature_dropout", 0.0),
            decoder_scale_init=cfg.model.get("decoder_scale_init", 5.0),
            decoder_hidden=cfg.model.get("decoder_hidden", cfg.model.hidden_channels),
        )

    raise ValueError(f"Unknown model.kind: {model_kind}")
