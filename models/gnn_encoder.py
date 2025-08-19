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
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data (torch_geometric.data.Data): 包含 x, edge_index, batch 的图数据对象。
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # 使用全局平均池化来获得整个图的表示
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding


def networkx_to_pyg_data(g: nx.Graph) -> Data:
    """
    将一个NetworkX图转换为PyTorch Geometric的Data对象。
    """
    if not g.nodes:
        # 处理空图的情况
        return Data(x=torch.empty((0, 1), dtype=torch.float32), edge_index=torch.empty((2, 0), dtype=torch.long))

    # 使用节点度作为简单的节点特征
    node_features = []
    node_mapping = {node: i for i, node in enumerate(g.nodes())}
    for node in g.nodes():
        node_features.append([g.degree(node)])

    x = torch.tensor(node_features, dtype=torch.float32)

    # 构建边索引
    edge_index = torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()
    # PyG需要双向边
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return Data(x=x, edge_index=edge_index)