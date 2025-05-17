import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dense_to_sparse

class RFGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=256, use_dynamic_graph=True):
        super(RFGCN, self).__init__()
        self.use_dynamic_graph = use_dynamic_graph

        # === 动态构图模块（带输入投影） ===
        self.feature_proj = torch.nn.Linear(num_features, 16)
        self.dynamic_attn = torch.nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)

        # 编码器层
        self.conv1 = GATConv(num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)

        # RSSI预测分支
        self.rssi_conv1 = GCNConv(hidden_channels, hidden_channels)
        self.rssi_conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.rssi_attention = torch.nn.Linear(hidden_channels//2, 1)
        self.rssi_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, hidden_channels//2),
            torch.nn.LayerNorm(hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(hidden_channels//2, hidden_channels//4),
            torch.nn.LayerNorm(hidden_channels//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(hidden_channels//4, 1)
        )

        # CQI预测分支
        self.cqi_conv1 = GCNConv(hidden_channels, hidden_channels)
        self.cqi_conv2 = GCNConv(hidden_channels, hidden_channels//2)

        channel_reduction = 2
        self.cqi_channel_attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, hidden_channels//channel_reduction),
            torch.nn.LayerNorm(hidden_channels//channel_reduction),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden_channels//channel_reduction, hidden_channels//2),
            torch.nn.LayerNorm(hidden_channels//2),
            torch.nn.Sigmoid()
        )

        self.cqi_spatial_attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, hidden_channels//4),
            torch.nn.LayerNorm(hidden_channels//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden_channels//4, hidden_channels//8),
            torch.nn.LayerNorm(hidden_channels//8),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels//8, 1),
            torch.nn.Sigmoid()
        )

        self.cqi_feature_fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels//2),
            torch.nn.LayerNorm(hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(hidden_channels//2, hidden_channels//2),
            torch.nn.LayerNorm(hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1)
        )

        self.cqi_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, hidden_channels//2),
            torch.nn.LayerNorm(hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden_channels//2, hidden_channels//4),
            torch.nn.LayerNorm(hidden_channels//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(hidden_channels//4, 1)
        )

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn_rssi1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn_rssi2 = torch.nn.BatchNorm1d(hidden_channels//2)
        self.bn_cqi1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn_cqi2 = torch.nn.BatchNorm1d(hidden_channels//2)

    def apply_attention(self, x, attention):
        weights = F.softmax(attention(x), dim=0)
        return x * weights

    def apply_cqi_attention(self, x):
        channel_weights = self.cqi_channel_attention(x)
        channel_out = x * channel_weights * 1.1
        spatial_weights = self.cqi_spatial_attention(x)
        spatial_out = x * spatial_weights * 0.9
        combined_features = torch.cat([channel_out, spatial_out], dim=-1)
        fused_features = self.cqi_feature_fusion(combined_features)
        return fused_features + x

    def build_dynamic_graph(self, x):
        x_proj = self.feature_proj(x)
        attn_output, attn_weights = self.dynamic_attn(x_proj.unsqueeze(0), x_proj.unsqueeze(0), x_proj.unsqueeze(0))
        attn_weights = attn_weights[0]

        k = 5  # top-k 注意力连接
        topk = torch.topk(attn_weights, k=k, dim=-1)
        row_idx = torch.arange(attn_weights.size(0)).unsqueeze(1).expand(-1, k).flatten()
        col_idx = topk.indices.flatten()
        edge_index = torch.stack([row_idx, col_idx], dim=0)
        return edge_index

    def forward(self, x, edge_index=None):
        if self.use_dynamic_graph or edge_index is None:
            edge_index = self.build_dynamic_graph(x)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.15, training=self.training)

        x_res = self.conv2(x, edge_index)
        x_res = self.bn2(x_res)
        x = F.relu(x_res + x)
        x = F.dropout(x, p=0.15, training=self.training)

        x_res = self.conv3(x, edge_index)
        x_res = self.bn3(x_res)
        x = F.relu(x_res + x)
        x = F.dropout(x, p=0.15, training=self.training)

        rssi = self.rssi_conv1(x, edge_index)
        rssi = self.bn_rssi1(rssi)
        rssi = F.relu(rssi)
        rssi = F.dropout(rssi, p=0.15, training=self.training)
        rssi = self.rssi_conv2(rssi, edge_index)
        rssi = self.bn_rssi2(rssi)
        rssi = F.relu(rssi)
        rssi = F.dropout(rssi, p=0.15, training=self.training)
        rssi = self.apply_attention(rssi, self.rssi_attention)
        rssi = self.rssi_pred(rssi)

        cqi = self.cqi_conv1(x, edge_index)
        cqi = self.bn_cqi1(cqi)
        cqi = F.relu(cqi)
        cqi = F.dropout(cqi, p=0.15, training=self.training)
        cqi = self.cqi_conv2(cqi, edge_index)
        cqi = self.bn_cqi2(cqi)
        cqi = F.relu(cqi)
        cqi = F.dropout(cqi, p=0.15, training=self.training)
        cqi = self.apply_cqi_attention(cqi)
        cqi = self.cqi_pred(cqi)

        return torch.cat([rssi, cqi], dim=1)
