# 文件: models/selector.py

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# 导入我们之前定义的GNN编码器
from .gnn_encoder import GNNEncoder


class SiameseGNNSelector(nn.Module):
    """
    一个孪生GNN模型，用于预测(芯片状态, 任务)对的匹配分数。
    """

    def __init__(self, chip_node_dim: int, task_node_dim: int, hidden_dim: int, embed_dim: int):
        super(SiameseGNNSelector, self).__init__()
        self.gnn_chip = GNNEncoder(chip_node_dim, hidden_dim, embed_dim)
        self.gnn_task = GNNEncoder(task_node_dim, hidden_dim, embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出一个[0, 1]之间的匹配分数
        )

    def forward(self, chip_data: Data, task_data: Data) -> torch.Tensor:
        # PyG的Batch对象可以自动处理批次
        chip_embedding = self.gnn_chip(chip_data)
        task_embedding = self.gnn_task(task_data)

        combined = torch.cat([chip_embedding, task_embedding], dim=1)
        return self.mlp_head(combined)
