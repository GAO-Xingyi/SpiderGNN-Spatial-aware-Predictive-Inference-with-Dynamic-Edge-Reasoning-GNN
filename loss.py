import torch
import torch.nn.functional as F

def custom_loss(pred, target):
    # 分离RSSI和CQI预测
    rssi_pred = pred[:, 0]
    cqi_pred = pred[:, 1]
    rssi_true = target[:, 0]
    cqi_true = target[:, 1]
    
    # RSSI损失 - 添加focal loss组件
    rssi_mse = F.mse_loss(rssi_pred, rssi_true)
    rssi_huber = F.huber_loss(rssi_pred, rssi_true, delta=1.0)
    rssi_focal = torch.mean((torch.abs(rssi_true - rssi_pred) ** 2) * torch.exp(-torch.abs(rssi_true)))
    rssi_loss = 0.6 * rssi_mse + 0.3 * rssi_huber + 0.1 * rssi_focal
    
    # CQI损失 - 添加自适应权重
    cqi_mse = F.mse_loss(cqi_pred, cqi_true)
    cqi_l1 = F.l1_loss(cqi_pred, cqi_true)
    # 对较大的CQI值赋予更高的权重
    cqi_weights = 1.0 + torch.abs(cqi_true) / torch.max(torch.abs(cqi_true))
    cqi_weighted = torch.mean(torch.abs(cqi_true - cqi_pred) * cqi_weights)
    
    # 根据预测难度动态调整权重
    cqi_difficulty = torch.exp(torch.abs(cqi_true - cqi_pred).mean())
    cqi_alpha = torch.clamp(0.4 + 0.1 * cqi_difficulty, min=0.4, max=0.6)
    
    cqi_loss = (1 - cqi_alpha) * cqi_mse + 0.3 * cqi_l1 + cqi_alpha * cqi_weighted
    
    # 总损失 - 使用动态权重
    rssi_weight = 0.55  # 基础权重
    cqi_weight = 0.45   # 基础权重
    
    # 根据批次中的异常值调整权重
    rssi_outliers = torch.sum(torch.abs(rssi_true - rssi_pred) > 2.0 * torch.std(rssi_true))
    cqi_outliers = torch.sum(torch.abs(cqi_true - cqi_pred) > 2.0 * torch.std(cqi_true))
    
    if rssi_outliers > 0:
        rssi_weight += 0.05
    if cqi_outliers > 0:
        cqi_weight += 0.05
        
    # 重新归一化权重
    total_weight = rssi_weight + cqi_weight
    rssi_weight = rssi_weight / total_weight
    cqi_weight = cqi_weight / total_weight
    
    total_loss = rssi_weight * rssi_loss + cqi_weight * cqi_loss
    
    return total_loss