import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

class RFGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=256):
        super(RFGCN, self).__init__()
        # 编码器层
        '''
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        '''
        self.conv1 = GATConv(num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        # self.conv4 = GATConv(num_features, hidden_channels)
        # self.conv5 = GATConv(hidden_channels, hidden_channels)
        # self.conv6 = GATConv(hidden_channels, hidden_channels)
        
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
        
        # CQI通道注意力
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
        
        # CQI空间注意力
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
        
        # CQI特征融合
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
        
        # CQI预测层
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
        
        # Batch Normalization层
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
        # 通道注意力
        channel_weights = self.cqi_channel_attention(x)
        channel_out = x * channel_weights * 1.1
        
        # 空间注意力
        spatial_weights = self.cqi_spatial_attention(x)
        spatial_out = x * spatial_weights * 0.9
        
        # 特征融合
        combined_features = torch.cat([channel_out, spatial_out], dim=-1)
        fused_features = self.cqi_feature_fusion(combined_features)
        
        # 残差连接
        fused_features = fused_features + x
        
        return fused_features
        
    def forward(self, x, edge_index):
        # 编码器层with残差连接
        identity = x
        
        # 第一层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.15, training=self.training)
        
        # 第二层 + 残差
        x_res = self.conv2(x, edge_index)
        x_res = self.bn2(x_res)
        x = F.relu(x_res + x)
        x = F.dropout(x, p=0.15, training=self.training)
        
        # 第三层
        x_res = self.conv3(x, edge_index)
        x_res = self.bn3(x_res)
        x = F.relu(x_res + x)
        x = F.dropout(x, p=0.15, training=self.training)
        
        # RSSI预测分支
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
        
        # CQI预测分支
        cqi = self.cqi_conv1(x, edge_index)
        cqi = self.bn_cqi1(cqi)
        cqi = F.relu(cqi)
        cqi = F.dropout(cqi, p=0.15, training=self.training)
        
        cqi = self.cqi_conv2(cqi, edge_index)
        cqi = self.bn_cqi2(cqi)
        cqi = F.relu(cqi)
        cqi = F.dropout(cqi, p=0.15, training=self.training)
        
        # 应用CQI注意力机制
        cqi = self.apply_cqi_attention(cqi)
        cqi = self.cqi_pred(cqi)
        
        return torch.cat([rssi, cqi], dim=1) 