import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def analyze_training_runs(best_metrics, train_histories):
    # 创建性能分析图表
    plt.figure(figsize=(15, 10))
    
    # 1. 验证损失比较
    plt.subplot(2, 2, 1)
    val_losses = [m['val_loss'] for m in best_metrics]
    plt.bar(range(1, len(val_losses) + 1), val_losses)
    plt.title('验证损失比较')
    plt.xlabel('训练运行')
    plt.ylabel('验证损失')
    
    # 2. RSSI和CQI的R2分数比较
    plt.subplot(2, 2, 2)
    rssi_r2 = [m['metrics']['RSSI']['R2'] for m in best_metrics]
    cqi_r2 = [m['metrics']['CQI']['R2'] for m in best_metrics]
    x = range(1, len(rssi_r2) + 1)
    plt.plot(x, rssi_r2, 'b-', label='RSSI R2')
    plt.plot(x, cqi_r2, 'r-', label='CQI R2')
    plt.title('R2分数比较')
    plt.xlabel('训练运行')
    plt.ylabel('R2分数')
    plt.legend()
    
    # 3. 训练过程分析
    plt.subplot(2, 2, 3)
    for run, history in train_histories.items():
        plt.plot(history['train_loss'], label=f'运行{run+1}')
    plt.title('训练损失比较')
    plt.xlabel('Epoch')
    plt.ylabel('训练损失')
    plt.legend()
    
    # 4. 验证过程分析
    plt.subplot(2, 2, 4)
    for run, history in train_histories.items():
        plt.plot(history['val_loss'], label=f'运行{run+1}')
    plt.title('验证损失比较')
    plt.xlabel('Epoch')
    plt.ylabel('验证损失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_analysis.png')
    plt.close()
    
    # 创建相关性分析
    metrics_df = pd.DataFrame([
        {
            'Run': i+1,
            'Val_Loss': m['val_loss'],
            'RSSI_MSE': m['metrics']['RSSI']['MSE'],
            'RSSI_R2': m['metrics']['RSSI']['R2'],
            'CQI_MSE': m['metrics']['CQI']['MSE'],
            'CQI_R2': m['metrics']['CQI']['R2'],
            'Early_Stop_Epoch': len(train_histories[i]['train_loss'])
        }
        for i, m in enumerate(best_metrics)
    ])
    
    # 计算指标间的相关性
    correlation = metrics_df.corr()
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('性能指标相关性分析')
    plt.tight_layout()
    plt.savefig('metrics_correlation.png')
    plt.close()
    
    return metrics_df

def select_best_ensemble_method(results_dict, true_values, pred_values_dict):
    """
    根据 R² 平均值选择最佳集成方法，并返回预测结果

    参数：
    - results_dict: dict，evaluate_ensemble_methods 返回的指标字典
    - true_values: ndarray，真实标签（经过反标准化）
    - pred_values_dict: dict，每种方法对应的预测结果

    返回：
    - best_method: str，最佳集成方法名
    - y_true: ndarray，真实标签
    - y_pred: ndarray，最佳方法的预测值
    - best_r2_avg: float，该方法的平均 R² 分数
    """
    best_method = None
    best_r2_avg = -np.inf

    for method, metrics in results_dict.items():
        r2_avg = np.mean([metrics["RSSI"]["R2"], metrics["CQI"]["R2"]])
        if r2_avg > best_r2_avg:
            best_r2_avg = r2_avg
            best_method = method

    y_true = true_values
    y_pred = pred_values_dict[best_method]

    return best_method, y_true, y_pred, best_r2_avg
