import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, test_loader, processor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # 反标准化预测值和真实值
    y_true = processor.inverse_transform_targets(y_true)
    y_pred = processor.inverse_transform_targets(y_pred)
    
    # 计算每个目标变量的评估指标
    metrics = {}
    target_names = ['RSSI', 'CQI']
    for i, name in enumerate(target_names):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2}
    
    return metrics, y_true, y_pred

def plot_predictions(y_true, y_pred):
    target_names = ['RSSI', 'CQI']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 确保输入数据是正确的格式
    y_true = np.array(y_true)
    if isinstance(y_pred, dict):
        y_pred = np.array([metrics['MSE'] for metrics in y_pred.values()])
    else:
        y_pred = np.array(y_pred)
    
    # 确保数据是2维的
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    for i, (name, ax) in enumerate(zip(target_names, axes)):
        if i < y_true.shape[1] and i < y_pred.shape[1]:
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]
            
            ax.scatter(true_vals, pred_vals, alpha=0.5)
            
            # 计算并绘制对角线
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel(f'真实 {name}')
            ax.set_ylabel(f'预测 {name}')
            ax.set_title(f'{name} 预测结果')
            
            # 添加R2分数
            r2 = r2_score(true_vals, pred_vals)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()