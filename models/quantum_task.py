import networkx as nx

class QuantumTask:
    """代表一个量子线路任务"""

    def __init__(self, task_id: int, num_qubits: int, depth: int, shots: int,
                 interaction_graph: nx.Graph, estimated_duration: float):
        self.id = task_id
        self.is_scheduled = False
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.interaction_graph = interaction_graph
        self.estimated_duration = estimated_duration

    def _generate_random_graph(self) -> nx.Graph:
        # 这是一个占位符，您需要用MQBench生成的线路来构建真实的交互图
        g = nx.gnp_random_graph(self.num_qubits, 0.6)
        return g

    def _estimate_duration(self) -> float:
        # 这是一个占位符，需要更真实的估算模型
        # 简化模型：时长 = 深度 * (平均门时间) * shots
        avg_gate_time = 20  # ns
        return self.depth * avg_gate_time * self.shots / 1e3  # 假设结果是微秒

    def reset(self):
        self.is_scheduled = False
