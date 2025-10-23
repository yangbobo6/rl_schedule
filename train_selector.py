# 文件: train_selector.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, random_split
import pickle
from tqdm import tqdm

from models.gnn_encoder import GNNEncoder


class SiameseGNNSelector(nn.Module):
    """
    一个孪生GNN模型，用于预测(芯片状态, 任务)对的匹配分数。
    """

    def __init__(self, chip_node_dim, task_node_dim, hidden_dim, embed_dim):
        super(SiameseGNNSelector, self).__init__()
        self.gnn_chip = GNNEncoder(chip_node_dim, hidden_dim, embed_dim)
        self.gnn_task = GNNEncoder(task_node_dim, hidden_dim, embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 将输出压缩到[0, 1]区间，匹配我们的标签
        )

    def forward(self, chip_data: Data, task_data: Data) -> torch.Tensor:
        chip_embedding = self.gnn_chip(chip_data)
        task_embedding = self.gnn_task(task_data)

        combined = torch.cat([chip_embedding, task_embedding], dim=1)
        return self.mlp_head(combined)


def pyg_collate_fn(batch):
    """自定义的collate_fn，用于将(芯片, 任务, 标签)的batch打包"""
    chip_graphs, task_graphs, labels = zip(*batch)

    chip_batch = Batch.from_data_list(list(chip_graphs))
    task_batch = Batch.from_data_list(list(task_graphs))
    label_batch = torch.stack(list(labels), 0)

    return chip_batch, task_batch, label_batch


def train_selector(dataset_path: str, model_save_path: str, epochs: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据集
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=pyg_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=pyg_collate_fn)

    # 2. 初始化模型
    model = SiameseGNNSelector(
        chip_node_dim=1, task_node_dim=1,  # 我们的节点特征维度都是1
        hidden_dim=64, embed_dim=32
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. 训练循环
    print("Starting GNN Selector pre-training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for chip_batch, task_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            chip_batch, task_batch, label_batch = chip_batch.to(device), task_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()
            predictions = model(chip_batch, task_batch)
            loss = criterion(predictions, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_loader):.4f}")

        # 验证... (省略以保持简洁)

    # 4. 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"GNN Selector model saved to {model_save_path}")


if __name__ == '__main__':
    # 运行此脚本来训练GNN选择器
    train_selector(
        dataset_path="data/selector_pretrain_dataset.pkl",
        model_save_path="models/gnn_selector_v0.pth"
    )
