# 文件: models/gnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import numpy as np


class GNNEncoder(nn.Module):
    """
    使用图卷积网络 (GCN) 将任务的交互图编码为固定长度的嵌入向量。
    """

    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int):
        super(GNNEncoder, self).__init__()
        # GCNConv 支持处理带权重的图
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 获取边权重，如果数据中没有(例如旧数据)，则设为None
        edge_weight = getattr(data, 'edge_weight', None)

        # 传入 edge_weight
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        return global_mean_pool(x, batch)


def networkx_to_pyg_data(g: nx.Graph) -> Data:
    """
    将一个NetworkX图转换为PyTorch Geometric的Data对象。
    """
    # 1. 处理空图的情况
    if not g.nodes:
        # 假设我们现在的节点特征维度是 2 (degree, gate_count)
        return Data(x=torch.empty((0, 2), dtype=torch.float32),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_weight=torch.empty((0,), dtype=torch.float32))

    # --- 1. 提取更丰富的节点特征 ---
    node_features = []
    # 归一化因子 (简单处理，防止数值过大)
    max_degree = len(g.nodes)
    # 获取最大门数，防止除以0
    gate_counts = [d for n, d in g.nodes(data='gate_count')]
    max_gates = max(gate_counts) if gate_counts else 1

    for node in g.nodes():
        degree = g.degree(node)
        gate_count = g.nodes[node].get("gate_count", 0)
        # 归一化特征 [degree_norm, gate_count_norm]
        node_features.append([degree / max_degree, gate_count / max_gates])

    x = torch.tensor(node_features, dtype=torch.float32)  # Shape: [NumNodes, 2]

    # --- 2. 提取边索引和边权重 ---
    edge_index = []
    edge_weights = []

    # 归一化因子
    weights = [d.get('weight', 1.0) for u, v, d in g.edges(data=True)]
    max_weight = max(weights) if weights else 1.0

    for u, v, data in g.edges(data=True):
        weight = data.get('weight', 1.0) / max_weight

        # 添加双向边 (PyG通常需要无向图的双向边)
        edge_index.append([u, v])
        edge_weights.append(weight)
        edge_index.append([v, u])
        edge_weights.append(weight)

    if not edge_index: # 处理有节点但无边的情况
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)  # Shape: [NumEdges]

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
