# 文件: create_dataset.py

import numpy as np
import random
from tqdm import tqdm
import pickle

from environment import QuantumSchedulingEnv
from heuristic_mapper import simple_greedy_mapper
from models.gnn_encoder import networkx_to_pyg_data


def create_pretraining_dataset(num_samples: int, save_path: str):
    """
    生成用于预训练GNN选择器的数据集。
    """
    dataset = []

    # 初始化一个临时的环境和任务生成器
    # 我们只需要它们的对象和方法，不需要进行RL训练
    print("Initializing environment to generate data...")

    # 创建一个虚拟的gnn_model和device，因为TaskGenerator需要它们
    class DummyGNN:
        def __init__(self, output_dim=16):
            self.output_dim = output_dim
            # 模仿真实GNN的结构，以便环境可以查询它
            self.conv2 = type('obj', (object,), {'out_channels': output_dim})

        def __call__(self, pyg_batch):
            # 这是一个假的forward方法
            # 它接收一个PyG的Batch对象
            # 并返回一个形状为 (batch_size, output_dim) 的全零张量
            batch_size = pyg_batch.num_graphs
            return torch.zeros((batch_size, self.output_dim))

    gate_times = {
        'u3': 50, 'u': 50, 'p': 30, 'cx': 350, 'id': 30,
        'measure': 1000, 'rz': 30, 'sx': 40, 'x': 40
    }

    dummy_gnn = DummyGNN(output_dim=16)

    env = QuantumSchedulingEnv(
        chip_size=(6, 6), num_tasks=20, max_qubits_per_task=8,
        gnn_model=dummy_gnn, device='cpu',
        gate_times=gate_times, reward_weights={}
    )

    # 从TaskGenerator的大任务池中获取任务
    all_tasks = list(env.task_generator.large_task_pool.values())

    print(f"Generating {num_samples} data samples...")
    for _ in tqdm(range(num_samples)):
        # 1. 生成一个随机的芯片状态
        placement_mask = np.zeros(env.num_qubits, dtype=np.int8)
        # 随机占用一些比特
        num_occupied = random.randint(0, env.num_qubits - 8)
        occupied_indices = np.random.choice(env.num_qubits, num_occupied, replace=False)
        placement_mask[occupied_indices] = 1

        # 2. 随机选择一个任务
        task = random.choice(all_tasks)

        # 3. 计算标签 (Y)
        estimated_swaps = simple_greedy_mapper(
            task.interaction_graph,
            env.chip_model,
            placement_mask,
            env.qubit_id_to_idx
        )
        label = 1.0 / (1.0 + float(estimated_swaps))

        # 4. 将输入 (X) 转换为 PyG Data 对象
        # a. 芯片状态图
        # 为了简化，我们只使用“是否被占用”作为节点特征
        chip_node_features = torch.tensor(placement_mask, dtype=torch.float32).unsqueeze(1)
        # 芯片的边索引是固定的
        chip_edge_index = []
        for qid, q in env.chip_model.items():
            for neighbor_id in q.connectivity:
                chip_edge_index.append([env.qubit_id_to_idx[qid], env.qubit_id_to_idx[neighbor_id]])
        chip_edge_index = torch.tensor(chip_edge_index, dtype=torch.long).t().contiguous()
        chip_graph_data = Data(x=chip_node_features, edge_index=chip_edge_index)

        # b. 任务交互图
        task_graph_data = networkx_to_pyg_data(task.interaction_graph)

        # 5. 存储数据点
        dataset.append((chip_graph_data, task_graph_data, torch.tensor([label], dtype=torch.float32)))

    # 保存数据集
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset with {len(dataset)} samples saved to {save_path}")


if __name__ == '__main__':
    import torch
    from torch_geometric.data import Data

    # 运行此脚本来创建数据集
    create_pretraining_dataset(num_samples=10000, save_path="data/selector_pretrain_dataset.pkl")
