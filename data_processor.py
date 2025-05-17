import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class RFDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def load_data(self):
        """加载Excel数据"""
        df = pd.read_excel(self.data_path)
        return df
    
    def create_graph(self, df, k=5):
        """
        创建图结构
        k: 每个节点连接的最近邻数量
        """
        # 提取位置信息作为节点坐标
        coords = df[['X', 'Y', 'Altitude']].values
        
        
        # 计算节点间的欧氏距离
        distances = np.zeros((len(coords), len(coords)))
        for i in range(len(coords)):
            for j in range(len(coords)):
                distances[i,j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
        
        
        '''
        
        # 基于信道相似度与位置的混合建图
        alpha = 0.5  # 控制信道/空间比重

        space_coords = df[['LAT', 'LON', 'Altitude']].values
        space_dist = np.linalg.norm(space_coords[:, None, :] - space_coords[None, :, :], axis=2)

        signal_feats = df[['RSSI_ext', 'RSRP_ext', 'RSRQ_ext', 'SNR_ext']].values
        signal_dist = np.linalg.norm(signal_feats[:, None, :] - signal_feats[None, :, :], axis=2)

        distances = alpha * signal_dist + (1 - alpha) * space_dist
        '''

        # 为每个节点找到k个最近邻
        edge_index = []
        for i in range(len(coords)):
            nearest = np.argsort(distances[i])[1:k+1]
            for j in nearest:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 添加双向边
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # 准备节点特征
        feature_columns = ['RSSI_ext', 'RSRP_ext', 'RSRQ_ext', 'SNR_ext', 
                         'TA', 'X', 'Y', 'Altitude','P_rc', 'Ave_Dist_Drone-bldg_core']
        features = df[feature_columns].values
        features = self.feature_scaler.fit_transform(features)
        
        ###---###
        # # ✴️ 信号类特征（保持原始尺度或单独归一化）
        # signal_features = df[['CQI_ext']].values

        # # ✅ 空间特征进行标准化
        # spatial_features = df[['RSSI_ext', 'RSRP_ext', 'RSRQ_ext', 'SNR_ext','X', 'Y', 'Altitude', 'TA', 'P_rc', 'Ave_Dist_Drone-bldg_core']].values
        # spatial_features = self.feature_scaler.fit_transform(spatial_features)

        # # 合并最终特征矩阵
        # features = np.concatenate([signal_features, spatial_features], axis=1)

        # # 目标变量标准化
        # targets = df[['RSSI_int', 'CQI_int']].values
        # targets = self.target_scaler.fit_transform(targets)
        ###---###

        # 准备目标变量并标准化
        targets = df[['RSSI_int', 'CQI_int']].values
        targets = self.target_scaler.fit_transform(targets)
        
        # 创建PyTorch Geometric数据对象
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(targets, dtype=torch.float)
        )
        
        return data
    
    def inverse_transform_targets(self, y):
        """将标准化的目标变量转换回原始尺度"""
        return self.target_scaler.inverse_transform(y)
    
    def process(self):
        """处理数据并返回图数据对象"""
        df = self.load_data()
        graph_data = self.create_graph(df)
        return graph_data

    def analyze_cqi_distribution(self):
        """分析CQI的分布特征"""
        df = self.load_data()
        cqi_values = df['CQI_int'].values
        
        # 创建图形
        plt.figure(figsize=(15, 5))
        
        # 1. CQI值分布直方图
        plt.subplot(131)
        sns.histplot(cqi_values, bins=15)
        plt.title('CQI Distribution')
        plt.xlabel('CQI Value')
        plt.ylabel('Count')
        
        # 2. CQI箱型图
        plt.subplot(132)
        sns.boxplot(y=cqi_values)
        plt.title('CQI Boxplot')
        plt.ylabel('CQI Value')
        
        # 3. CQI与RSSI的关系散点图
        plt.subplot(133)
        sns.scatterplot(data=df, x='RSSI_int', y='CQI_int', alpha=0.5)
        plt.title('CQI vs RSSI')
        plt.xlabel('RSSI')
        plt.ylabel('CQI')
        
        plt.tight_layout()
        plt.savefig('cqi_analysis.png')
        plt.show()
        
        # 打印统计信息
        print("\nCQI Statistical Analysis:")
        print(f"Unique values: {sorted(np.unique(cqi_values))}")
        print(f"Mean: {np.mean(cqi_values):.2f}")
        print(f"Std: {np.std(cqi_values):.2f}")
        print(f"Min: {np.min(cqi_values)}")
        print(f"Max: {np.max(cqi_values)}")
        print(f"25th percentile: {np.percentile(cqi_values, 25)}")
        print(f"Median: {np.median(cqi_values)}")
        print(f"75th percentile: {np.percentile(cqi_values, 75)}")
        
        return df['CQI_int'].value_counts().sort_index()

if __name__ == '__main__':
    # 测试数据处理
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = Path().resolve()
    data_path = os.path.join(current_dir, '数据集', 'Simulated Dataset for DM v1.xlsx')
    
    processor = RFDataProcessor(data_path)
    data = processor.process()
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of node features: {data.num_features}')
    print(f'Target shape: {data.y.shape}')
    
    cqi_distribution = processor.analyze_cqi_distribution()
    print("\nCQI Value Distribution:")
    print(cqi_distribution)     