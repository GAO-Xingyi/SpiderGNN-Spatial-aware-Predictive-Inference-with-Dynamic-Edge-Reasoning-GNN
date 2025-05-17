import torch
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from data_processor import RFDataProcessor
from ensemble import ModelEnsemble, evaluate_ensemble_methods
from evaluate import plot_predictions
import numpy as np
from utils.analysis import select_best_ensemble_method  
from evaluate import plot_predictions, plot_prediction_histograms  

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

    test_true = []
    test_pred = []
    for data in test_loader:
        data = data.to(device)
        pred = ensemble.predict(data, device, method='dynamic')
        test_true.append(data.y.cpu().numpy())
        test_pred.append(pred.numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_true = processor.inverse_transform_targets(test_true)
    test_pred = processor.inverse_transform_targets(test_pred)

    plot_predictions(test_true, test_pred)
 