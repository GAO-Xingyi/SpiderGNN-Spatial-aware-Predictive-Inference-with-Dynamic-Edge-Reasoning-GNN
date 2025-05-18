from model import MetaLearner
from rfgcn import RFGCN
from train import train_model
from evaluate import evaluate_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge


class EnsembleStrategy:
    @staticmethod
    def weighted_average(predictions, weights):
        """简单加权平均"""
        return np.average(predictions, axis=0, weights=weights)
    
    @staticmethod
    def dynamic_weighted_average(predictions, val_predictions, val_targets):
        """基于验证集性能的动态权重"""
        weights = np.zeros(len(predictions))
        
        for i in range(len(predictions)):
            rssi_mse = mean_squared_error(val_targets[:, 0], val_predictions[i][:, 0])
            cqi_mse = mean_squared_error(val_targets[:, 1], val_predictions[i][:, 1])
            weights[i] = 1.0 / (rssi_mse * 0.6 + cqi_mse * 0.4)
        
        weights = weights / weights.sum()
        return np.average(predictions, axis=0, weights=weights)
    
    @staticmethod
    def advanced_stacking(predictions, val_predictions, val_targets, test_predictions):
        """使用多种元学习器的堆叠集成"""
        meta_predictions = np.zeros_like(test_predictions[0])
        
        # 为RSSI和CQI选择不同的元学习器
        for i in range(2):  # 0: RSSI, 1: CQI
            # 准备数据
            meta_features = np.column_stack([pred[:, i] for pred in val_predictions])
            meta_targets = val_targets[:, i]
            test_meta_features = np.column_stack([pred[:, i] for pred in test_predictions])
            
            # 标准化特征
            scaler = StandardScaler()
            meta_features = scaler.fit_transform(meta_features)
            test_meta_features = scaler.transform(test_meta_features)
            
            # 初始化所有元学习器
            meta_learners = {
                'ridge': Ridge(alpha=0.5),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=250,           # 适度的树数量
                    learning_rate=0.004,        # 适中的学习率
                    max_depth=4,                # 降回原来的深度
                    min_child_weight=2,         # 降低最小子节点权重
                    subsample=0.9,              # 提高样本采样率
                    colsample_bytree=0.9,       # 提高特征采样率
                    reg_alpha=0.04,             # 适度的L1正则化
                    reg_lambda=0.05,            # 保持L2正则化
                    gamma=0.05,                 # 降低分裂阈值
                    random_state=42,
                    tree_method='exact',
                    booster='gbtree'
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.008,
                    num_leaves=31,
                    reg_alpha=0.05,
                    reg_lambda=0.05,
                    verbose_eval=-1
                ),
                'catboost': CatBoostRegressor(
                    iterations=200,
                    learning_rate=0.008,
                    depth=4,
                    verbose=False
                ),
                'nn': None  # 将在后面初始化
            }
            
            # 训练神经网络元学习器
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_meta = MetaLearner(meta_features.shape[1]).to(device)
            optimizer = torch.optim.Adam(nn_meta.parameters(), lr=0.0008, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            meta_features_tensor = torch.FloatTensor(meta_features).to(device)
            meta_targets_tensor = torch.FloatTensor(meta_targets).to(device)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(500):
                nn_meta.train()
                optimizer.zero_grad()
                output = nn_meta(meta_features_tensor)
                loss = F.mse_loss(output.squeeze(), meta_targets_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_meta.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 更新学习率
                scheduler.step(loss)
                
                # 早停检查
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        break
            
            meta_learners['nn'] = nn_meta
            
            # 评估每个元学习器的性能
            best_score = float('inf')
            best_learner = None
            best_predictions = None
            
            for name, learner in meta_learners.items():
                if name == 'nn':
                    learner.eval()
                    with torch.no_grad():
                        val_pred = learner(torch.FloatTensor(meta_features).to(device))
                        val_pred = val_pred.cpu().numpy().squeeze()
                        test_pred = learner(torch.FloatTensor(test_meta_features).to(device))
                        test_pred = test_pred.cpu().numpy().squeeze()
                else:
                    learner.fit(meta_features, meta_targets)
                    val_pred = learner.predict(meta_features)
                    test_pred = learner.predict(test_meta_features)
                
                score = mean_squared_error(meta_targets, val_pred)
                if score < best_score:
                    best_score = score
                    best_learner = name
                    best_predictions = test_pred
            
            print(f"{'RSSI' if i == 0 else 'CQI'} 最佳元学习器: {best_learner}")
            meta_predictions[:, i] = best_predictions
        
        return meta_predictions

class ModelEnsemble:
    def __init__(self, num_models=5):
        self.num_models = num_models
        self.models = []
        self.weights = None
        self.val_predictions = None
        self.val_targets = None
        self.strategy = EnsembleStrategy()
    
    def train_models(self, train_loader, val_loader, num_features, device, num_epochs=1000):
        val_losses = []
        self.val_predictions = []
        
        # 获取验证集目标值
        for data in val_loader:
            self.val_targets = data.y.cpu().numpy()
        
        for i in range(self.num_models):
            # 设置不同的随机种子
            seed = 42 + i * 100
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            
            print(f"\n训练模型 {i+1}/{self.num_models}")
            model = RFGCN(num_features=num_features).to(device)
            train_losses, val_loss = train_model(model, train_loader, val_loader, num_epochs)
            
            # 保存模型
            model_path = f'model_{i+1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, model_path)
            
            # 获取验证集预测
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    val_pred = model(data.x, data.edge_index)
                    self.val_predictions.append(val_pred.cpu().numpy())
            
            self.models.append(model)
            val_losses.append(val_loss)
        
        # 计算基础权重
        val_losses = np.array(val_losses)
        weights = np.exp(-val_losses)
        self.weights = weights / weights.sum()
        
        print("\n基础模型权重:", self.weights)
    
    def predict(self, data, device, method='advanced_stacking'):
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(data.x.to(device), data.edge_index.to(device))
                all_predictions.append(pred.cpu().numpy())
        
        # 根据选择的方法进行集成
        if method == 'weighted_average':
            ensemble_pred = self.strategy.weighted_average(all_predictions, self.weights)
        elif method == 'dynamic':
            ensemble_pred = self.strategy.dynamic_weighted_average(
                all_predictions, self.val_predictions, self.val_targets)
        elif method == 'advanced_stacking':
            ensemble_pred = self.strategy.advanced_stacking(
                all_predictions, self.val_predictions, self.val_targets, all_predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return torch.tensor(ensemble_pred)

def evaluate_ensemble_methods(ensemble, test_loader, processor, device):
    methods = ['weighted_average', 'dynamic', 'advanced_stacking']
    results = {}
    
    for method in methods:
        print(f"\n评估集成方法: {method}")
        ensemble_predictions = []
        true_values = []
        
        for data in test_loader:
            data = data.to(device)
            pred = ensemble.predict(data, device, method=method)
            ensemble_predictions.append(pred.numpy())
            true_values.append(data.y.cpu().numpy())
        
        ensemble_predictions = np.concatenate(ensemble_predictions)
        true_values = np.concatenate(true_values)
        
        # 反标准化预测值和真实值
        ensemble_predictions = processor.inverse_transform_targets(ensemble_predictions)
        true_values = processor.inverse_transform_targets(true_values)
        
        # 计算评估指标
        metrics = {}
        target_names = ['RSSI', 'CQI']
        for i, name in enumerate(target_names):
            mse = mean_squared_error(true_values[:, i], ensemble_predictions[:, i])
            r2 = r2_score(true_values[:, i], ensemble_predictions[:, i])
            metrics[name] = {'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2}
        
        results[method] = metrics
    
    return results