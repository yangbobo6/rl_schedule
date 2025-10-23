# 文件: heuristic_mapper.py

import networkx as nx
import numpy as np
from typing import Dict, Tuple


def simple_greedy_mapper(task_graph: nx.Graph,
                         chip_model: Dict[Tuple[int, int], 'PhysicalQubit'],
                         initial_placement_mask: np.ndarray,
                         qubit_id_to_idx: Dict) -> int:
    """
    一个简单的贪心启发式映射器，用于快速估算SWAP数量。

    Args:
        task_graph: 任务的交互图。
        chip_model: 芯片的物理模型。
        initial_placement_mask: 标记芯片上哪些比特已经被占用。
        qubit_id_to_idx: 物理比特ID到索引的映射。

    Returns:
        int: 估算的SWAP数量。
    """
    if not task_graph.nodes:
        return 0

    num_swaps = 0
    mapping = {}

    # 1. 对逻辑比特按度数从高到低排序
    sorted_logical_qubits = sorted(task_graph.degree, key=lambda x: x[1], reverse=True)

    available_physical_indices = [i for i, v in enumerate(initial_placement_mask) if v == 0]

    if len(available_physical_indices) < len(sorted_logical_qubits):
        return 1000  # 如果可用比特不足，返回一个巨大的SWAP数

    # 2. 贪心放置
    for logical_qubit_id, _ in sorted_logical_qubits:
        best_physical_idx = -1
        min_cost = float('inf')

        # 遍历所有可用的物理比特
        for physical_idx in available_physical_indices:
            cost = 0
            # 检查与已放置邻居的连通性
            for placed_logical, placed_physical_id in mapping.items():
                if task_graph.has_edge(logical_qubit_id, placed_logical):
                    physical_id = [k for k, v in qubit_id_to_idx.items() if v == physical_idx][0]
                    if physical_id not in chip_model[placed_physical_id].connectivity:
                        cost += 1  # 每有一个断开的连接，成本+1

            # 寻找成本最低的位置
            if cost < min_cost:
                min_cost = cost
                best_physical_idx = physical_idx

        if best_physical_idx != -1:
            mapping[logical_qubit_id] = [k for k, v in qubit_id_to_idx.items() if v == best_physical_idx][0]
            available_physical_indices.remove(best_physical_idx)
            num_swaps += min_cost
        else:
            # 如果没有找到任何可用位置（理论上不应发生）
            return 1000

    return num_swaps
