# 文件: model.py
import torch

class MetaLearner(torch.nn.Module):
    def __init__(self, input_size):
        super(MetaLearner, self).__init__()
        
        # 特征投影层
        self.input_proj = torch.nn.Linear(input_size, 256)
        
        # 主干网络层
        self.layers = torch.nn.ModuleList([
            self._make_layer(256, 192),
            self._make_layer(192, 128),
            self._make_layer(128, 64)   
           
            
        ])
        
        #self.input_proj = torch.nn.Linear(input_size, 128)
        #self.layers = torch.nn.ModuleList([
        #    self._make_layer(128, 96),
        #    self._make_layer(96, 64),
        #    self._make_layer(64, 32)
            
        #])
        
        # 4头注意力层
        self.attention = torch.nn.MultiheadAttention(256, num_heads=4, dropout=0.1, batch_first=True)
        
        #self.attention = torch.nn.MultiheadAttention(128, num_heads=4, dropout=0.1, batch_first=True)
        
        # 输出层
        self.output = torch.nn.Linear(64, 1)
        #self.output = torch.nn.Linear(32, 1)
        
        # 递减的Dropout
        self.dropouts = torch.nn.ModuleList([
            torch.nn.Dropout(0.2),
            torch.nn.Dropout(0.15),
            torch.nn.Dropout(0.1)            
        ])
        
        # 初始化权重
        self._init_weights()
    
    def _make_layer(self, in_features, out_features):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.BatchNorm1d(out_features),
            torch.nn.GELU(),
            torch.nn.LayerNorm(out_features)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    
    def _apply_attention(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        return attn_out.squeeze(1)
    
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # 应用注意力
        x = self._apply_attention(x)
        
        # 主干网络处理
        for layer, dropout in zip(self.layers, self.dropouts):
            residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + residual
            x = dropout(x)
        
        return self.output(x)
