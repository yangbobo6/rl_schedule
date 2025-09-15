# 文件: task_generator.py
import os

import torch
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit import transpile
from models.gnn_encoder import GNNEncoder, networkx_to_pyg_data
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
try:
    from mqbench.application.qft import QFT
except ImportError:
    print("Warning: MQBench not found or not configured. Using dummy circuits.")
    QFT = None

# 导入环境中的 QuantumTask 类以避免循环导入
from models.quantum_task import QuantumTask


class TaskGenerator:
    """
    使用MQBench生成量子线路，并将其解析为环境所需的QuantumTask对象。
    """

    def __init__(self, gate_times: dict, gnn_model: GNNEncoder, device):
        """
        初始化时，需要提供标准门的执行时间以估算任务时长。

        Args:
            gate_times (dict): 一个字典，包含门名称和它们的执行时间（例如，单位为ns）。
                               示例: {'cx': 300, 'u3': 50, 'measure': 1000}
        """
        self.gate_times = gate_times
        # 基础门列表现在只用于 transpile 函数
        self.basis_gates = list(gate_times.keys())
        self.gnn_model = gnn_model
        self.device = device
        self.large_task_pool = self.generate_tasks(100)  # 一次性生成100个不同的任务

    def get_episode_tasks(self, num_tasks: int) -> dict:
        # 从大池中随机采样
        sampled_ids = np.random.choice(list(self.large_task_pool.keys()), num_tasks, replace=False)
        episode_tasks = {i: self.large_task_pool[tid] for i, tid in enumerate(sampled_ids)}
        return episode_tasks


    def generate_tasks(self, num_tasks: int) -> dict:
        """
        生成指定数量的任务。
        """
        tasks = {}
        for i in range(num_tasks):
            num_qubits = np.random.randint(2, 6)

            if QFT:
                circuit = QFT(num_qubits)
            else:
                circuit = self._create_dummy_circuit(num_qubits)

            task = self._parse_circuit(i, circuit)
            tasks[i] = task
            if i < 5:
                # 确保 results 文件夹存在
                os.makedirs("results", exist_ok=True)
                # 保存为 PNG 文件
                fig = circuit.draw(output="mpl")
                filepath = os.path.join("results", f"dummy_circuit{i}.png")
                fig.savefig(filepath, dpi=300)
                plt.close(fig)

        return tasks

    def _parse_circuit(self, task_id: int, circuit: QuantumCircuit) -> 'QuantumTask':
        """
        解析一个Qiskit QuantumCircuit对象，提取所有需要的信息。
        """
        # --- a. 使用 transpile 函数分解线路 ---
        # transpile 是一个功能强大的函数，它会处理所有的转换逻辑
        # 我们告诉它我们的目标基础门集，它会返回一个转换好的线路
        decomposed_circuit = transpile(circuit, basis_gates=self.basis_gates, optimization_level=0)
        # optimization_level=0 确保它只做基础转换，不做复杂的优化，这样我们能得到更可预测的门数

        # --- b. 提取基本信息 ---
        num_qubits = decomposed_circuit.num_qubits
        depth = decomposed_circuit.depth()
        shots = 1024

        # --- c. 构建交互图 (Interaction Graph) ---
        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(range(num_qubits))

        for instruction in decomposed_circuit.data:
            q_indices = [decomposed_circuit.qubits.index(q) for q in instruction.qubits]

            if len(q_indices) > 1:
                from itertools import combinations
                for u, v in combinations(q_indices, 2):
                    interaction_graph.add_edge(u, v)
        if task_id<5:
            plt.figure()  # 创建一个新画布，这是一个好习惯
            nx.draw(interaction_graph, with_labels=True, node_color='skyblue', edge_color='gray')
            # 构建一个唯一的文件名，避免文件被覆盖
            filename = f"interaction_graph_task_{task_id}.png"
            plt.savefig(filename)  # 将当前画布上的图像保存到文件
            plt.close()  # 关闭画布，释放内存，防止后续绘图混在一起

        # --- d. 计算估算执行时长 (Estimated Duration) ---
        gate_counts = decomposed_circuit.count_ops()
        duration = 0
        for gate_name, count in gate_counts.items():
            if gate_name in self.gate_times:
                duration += self.gate_times[gate_name] * count

        # 假设测量是最后一步，并且可以并行
        if 'measure' in self.gate_times:
            # 这里可能需要更精细的逻辑，但为了简化，我们只加一次测量时间
            pass  # 时长模型可能需要重新考虑，但我们先让代码跑起来

        total_duration = duration * shots  # 假设总时长是门操作时长 * shots

        # 预计算GNN嵌入 ---
        pyg_data = networkx_to_pyg_data(interaction_graph)
        if pyg_data.x.shape[0] > 0:
            pyg_batch = Batch.from_data_list([pyg_data]).to(self.device)
            with torch.no_grad():
                graph_embedding = self.gnn_model(pyg_batch).cpu().numpy().flatten()
        else:
            graph_embedding = np.zeros(self.gnn_model.conv2.out_channels)  # GNN输出维度

        # --- e. 创建我们环境中的Task对象 ---
        task = QuantumTask(
            task_id=task_id,
            num_qubits=num_qubits,
            depth=depth,
            shots=shots,
            interaction_graph=interaction_graph,
            graph_embedding=graph_embedding,
            estimated_duration=total_duration / 1e3  # 转换为微秒 (us)
        )
        return task

    def _create_dummy_circuit(self, num_qubits: int) -> QuantumCircuit:
        """一个备用的虚拟线路生成器"""
        qc = QuantumCircuit(num_qubits)
        depth = np.random.randint(5, 15)
        for _ in range(depth):
            for i in range(num_qubits):
                if np.random.rand() > 0.5:
                    # 使用Qiskit推荐的 u 门替代 u3
                    qc.u(np.random.rand() * np.pi, np.random.rand() * np.pi, np.random.rand() * np.pi, i)
            if num_qubits > 1 and np.random.rand() > 0.3:
                q1, q2 = np.random.choice(num_qubits, 2, replace=False)
                qc.cx(q1, q2)
        qc.measure_all()
        return qc