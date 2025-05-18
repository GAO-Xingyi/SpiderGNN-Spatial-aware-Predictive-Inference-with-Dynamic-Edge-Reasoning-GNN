import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

def evaluate_model(model, test_loader, processor, output_path='model_evaluation_metrics.txt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    start_time = time.time()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    end_time = time.time()

    # Scoring time in seconds
    scoring_time = end_time - start_time

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Inverse transform targets
    y_true = processor.inverse_transform_targets(y_true)
    y_pred = processor.inverse_transform_targets(y_pred)

    metrics = {}
    target_names = ['RSSI', 'CQI']
    with open(output_path, 'w') as f:
        for i, name in enumerate(target_names):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            se = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            corr, _ = pearsonr(y_true[:, i], y_pred[:, i])

            metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Squared Error': se,
                'Correlation': corr,
                'Scoring Time (s)': scoring_time
            }

            f.write(f"{name} Metrics:\n")
            f.write(f"  RMSE: {rmse:.4f}\n")
            f.write(f"  MAE: {mae:.4f}\n")
            f.write(f"  R2: {r2:.4f}\n")
            f.write(f"  Squared Error: {se:.4f}\n")
            f.write(f"  Correlation: {corr:.4f}\n")
            f.write(f"  Scoring Time (s): {scoring_time:.2f}\n\n")

    return metrics, y_true, y_pred

def plot_predictions(y_true, y_pred):
    target_names = ['RSSI', 'CQI']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    for i, (name, ax) in enumerate(zip(target_names, axes)):
        if i < y_true.shape[1] and i < y_pred.shape[1]:
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]

            ax.scatter(true_vals, pred_vals, alpha=0.5)
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Prediction Results')

            r2 = r2_score(true_vals, pred_vals)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('scatter_prediction_results.png')
    plt.show()

def plot_prediction_histograms(y_true, y_pred):
    target_names = ['RSSI', 'CQI']
    colors = [('darkorange', 'royalblue'), ('darkorange', 'royalblue')]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 横向排列

    for i, (name, ax) in enumerate(zip(target_names, axes)):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        ax.hist(true_vals, bins=15, alpha=0.6, label=f"True {name}", color=colors[i][0], edgecolor='black')
        ax.hist(pred_vals, bins=15, alpha=0.6, label=f"Predicted {name}", color=colors[i][1], edgecolor='black')

        ax.set_xlabel(f"{name} Values", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Histogram of True vs. Predicted {name}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("best_method_prediction_histogram.png", dpi=300)
    plt.show()
