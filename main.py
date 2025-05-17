import torch
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from data_processor import RFDataProcessor
from ensemble import ModelEnsemble, evaluate_ensemble_methods
from evaluate import plot_predictions, plot_prediction_histograms, evaluate_model  # ✅ 加载评估函数
from utils.analysis import select_best_ensemble_method
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def save_extended_metrics_to_txt(y_true, y_pred, output_path='best_model_metrics.txt'):
    """
    保存完整的模型评估指标到 TXT 文件，包括：
    MSE、RMSE、R²、MAE、Correlation。
    """
    target_names = ['RSSI', 'CQI']
    with open(output_path, 'w') as f:
        for i, name in enumerate(target_names):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            corr, _ = pearsonr(y_true[:, i], y_pred[:, i])

            f.write(f"{name} Metrics:\n")
            f.write(f"  MSE: {mse:.4f}\n")
            f.write(f"  RMSE: {rmse:.4f}\n")
            f.write(f"  R2: {r2:.4f}\n")
            f.write(f"  MAE: {mae:.4f}\n")
            f.write(f"  Correlation: {corr:.4f}\n\n")

    print(f"✅ Evaluation metrics written to {output_path}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = Path().resolve()
    data_path = "Algorithm implementation\dataset\Simulated Dataset for DM v1.xlsx"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据集路径: {data_path}")

    processor = RFDataProcessor(data_path)
    data = processor.process()

    indices = list(range(data.num_nodes))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    train_loader = DataLoader([data], batch_size=1)
    val_loader = DataLoader([data], batch_size=1)
    test_loader = DataLoader([data], batch_size=1)

    ensemble = ModelEnsemble(num_models=5)
    ensemble.train_models(train_loader, val_loader, data.num_features, device)

    results = evaluate_ensemble_methods(ensemble, test_loader, processor, device)

    for method, metrics in results.items():
        print(f"\n{method} 集成方法结果:")
        for target, metric in metrics.items():
            print(f'\n{target} Metrics:')
            for name, value in metric.items():
                print(f'{name}: {value:.4f}')

    # ✅ 收集所有方法的预测值
    pred_values_dict = {}
    test_true = []
    for data in test_loader:
        data = data.to(device)
        test_true.append(data.y.cpu().numpy())
        for method in results:
            pred = ensemble.predict(data, device, method=method)
            if method not in pred_values_dict:
                pred_values_dict[method] = []
            pred_values_dict[method].append(pred.cpu().numpy())

    test_true = np.concatenate(test_true)
    test_true = processor.inverse_transform_targets(test_true)

    for method in pred_values_dict:
        pred_values_dict[method] = np.concatenate(pred_values_dict[method])
        pred_values_dict[method] = processor.inverse_transform_targets(pred_values_dict[method])

    # ✅ 选择最佳集成方法
    best_method, y_true, y_pred, best_r2 = select_best_ensemble_method(results, test_true, pred_values_dict)
    print(f"\n✅ 最佳集成方法为: {best_method}，平均 R² = {best_r2:.4f}")

    # ✅ 使用最佳方法的结果进行绘图
    plot_predictions(y_true, y_pred)
    plot_prediction_histograms(y_true, y_pred)
    save_extended_metrics_to_txt(y_true, y_pred)

