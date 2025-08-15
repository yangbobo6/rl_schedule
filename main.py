import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple

from task_generate import TaskGenerator
from torch.utils.tensorboard import SummaryWriter # 导入TensorBoard的Writer
import time

# === 辅助数据类 ===
# 用于清晰地组织数据，方便管理

class PhysicalQubit:
    """代表一个物理量子比特及其所有属性"""

    def __init__(self, q_id: Tuple[int, int], t1: float, t2: float, f_1q: float, f_ro: float):
        self.id = q_id
        # 静态属性
        self.t1 = t1
        self.t2 = t2
        self.fidelity_1q = f_1q
        self.fidelity_readout = f_ro
        self.connectivity = {}  # 存储与邻居的连接信息
        # 动态属性
        self.booking_schedule = []  # 存储 (start_time, end_time)

    def add_link(self, neighbor_id: Tuple[int, int], f_2q: float, crosstalk: float):
        self.connectivity[neighbor_id] = {'fidelity_2q': f_2q, 'crosstalk_coeff': crosstalk}

    def get_next_available_time(self) -> float:
        if not self.booking_schedule:
            return 0.0
        return self.booking_schedule[-1][1]  # 假设预定是按时间排序的

    def reset(self):
        self.booking_schedule = []


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


# === 主环境类 ===

class QuantumSchedulingEnv(gym.Env):
    """
    用于量子电路离线调度的Gymnasium环境。
    智能体的任务是为一系列量子任务生成一个完整的调度方案。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, chip_size: Tuple[int, int] = (5, 5), num_tasks: int = 10, max_qubits_per_task: int = 6):
        super().__init__()

        self.chip_size = chip_size
        self.num_qubits = chip_size[0] * chip_size[1]
        self.num_tasks = num_tasks
        self.max_qubits_per_task = max_qubits_per_task  # 用于定义观察空间

        # 定义一个假设的门时间表 (单位: ns)
        self.gate_times = {
            'u3': 50,  # 代表所有单比特门
            'u2': 40,
            'u1': 30,
            'cx': 350,  # 控制非门
            'id': 30,  # 空闲门
            'measure': 1000,  # 测量
        }
        # --- GNN编码器 (用于将任务交互图编码为向量) ---
        # 在实际项目中，这会是一个预训练或端到端训练的PyTorch/TensorFlow模型
        # 这里我们用一个虚拟的实现代替
        # self.gnn_encoder = GNNEncoder(embedding_dim=16

        # --- 核心数据结构 ---
        self.task_generator = TaskGenerator(self.gate_times)
        self.chip_model: Dict[Tuple[int, int], PhysicalQubit] = self._create_chip_model()
        self.task_pool: Dict[int, QuantumTask] = self._create_task_pool()
        self.schedule_plan: List[Dict] = []
        self.current_step = 0

        # --- 预计算用于归一化的值 ---
        self._precompute_normalization_factors()

        # 建立物理比特ID到索引的映射
        self.qubit_id_to_idx = {qid: i for i, qid in enumerate(sorted(self.chip_model.keys()))}
        self.idx_to_qubit_id = {i: qid for qid, i in self.qubit_id_to_idx.items()}

        # --- 定义动作和观察空间 (这是个难点，先用简化版占位) ---
        # 这是一个复杂的混合动作空间，实际应用中需要用更复杂的库或自定义实现
        # 这里我们先定义一个概念上的结构
        # 动作: (任务ID, [物理比特ID列表], 开始时间)
        # 观察: 字典，包含芯片状态和任务状态
        # 为了简化，我们暂时使用Box空间作为占位符，实际需要自定义
        self.action_space = spaces.Dict({
            "task_id": spaces.Discrete(self.num_tasks),
            # 映射和时间非常复杂，这里先不定义，在step中直接接收
        })

        # --- 定义详细的观察空间 (Observation Space) ---
        # 芯片嵌入维度: f_1q, f_ro, t1, t2, x, y, next_avail_time (7)
        #             + 4 * (f_2q, crosstalk) for neighbors (8) -> Total 15
        D_QUBIT = 15
        # 任务标量维度: num_q, depth, duration, shots, num_edges, mean_degree
        D_TASK_SCALAR = 6
        # GNN嵌入维度 (占位符)
        D_GRAPH_EMBED = 0  # 暂时不使用单独的GNN嵌入
        D_TASK = D_TASK_SCALAR + D_GRAPH_EMBED

        self.observation_space = spaces.Dict({
            # 芯片状态: 25个比特，每个有D_QUBIT个特征
            "qubit_embeddings": spaces.Box(low=-1, high=1, shape=(self.num_qubits, D_QUBIT), dtype=np.float32),

            # 当前要调度的任务的状态: 1个任务，有D_TASK个特征
            "current_task_embedding": spaces.Box(low=-1, high=1, shape=(D_TASK,), dtype=np.float32),

            # 已放置比特的掩码 (用于自回归放置)
            # 0表示未选, 1表示已选
            "placement_mask": spaces.Box(low=0, high=1, shape=(self.num_qubits,), dtype=np.int8),

            # 逻辑比特的上下文: 当前逻辑比特索引, 它与已放置比特的连接数
            "logical_qubit_context": spaces.Box(low=0, high=self.max_qubits_per_task, shape=(2,), dtype=np.float32)
        })

    def _create_chip_model(self) -> Dict[Tuple[int, int], PhysicalQubit]:
        """创建并初始化芯片模型，包括模拟的物理属性"""
        chip = {}
        # 模拟属性的分布
        for r in range(self.chip_size[0]):
            for c in range(self.chip_size[1]):
                q_id = (r, c)
                # 模拟中心比特质量更高
                center_dist = np.sqrt((r - self.chip_size[0] / 2) ** 2 + (c - self.chip_size[1] / 2) ** 2)
                quality_factor = np.exp(-center_dist / 3.0)  # 0 to 1

                t1 = 80 + 40 * quality_factor * np.random.uniform(0.9, 1.1)  # us
                t2 = 60 + 50 * quality_factor * np.random.uniform(0.9, 1.1)  # us
                f_1q = 0.999 - 0.001 * (1 - quality_factor) * np.random.uniform(0.9, 1.1)
                f_ro = 0.98 - 0.02 * (1 - quality_factor) * np.random.uniform(0.9, 1.1)

                chip[q_id] = PhysicalQubit(q_id, t1, t2, f_1q, f_ro)

        # 创建连接性
        for r in range(self.chip_size[0]):
            for c in range(self.chip_size[1]):
                q_id = (r, c)
                for dr, dc in [(0, 1), (1, 0)]:  # 只需检查右边和下边
                    neighbor_id = (r + dr, c + dc)
                    if neighbor_id in chip:
                        # 模拟连接质量
                        f_2q = 0.99 - 0.02 * np.random.random()
                        crosstalk = 0.05 + 0.1 * np.random.random()
                        chip[q_id].add_link(neighbor_id, f_2q, crosstalk)
                        chip[neighbor_id].add_link(q_id, f_2q, crosstalk)  # 双向连接
        return chip

    def _create_task_pool(self) -> Dict[int, QuantumTask]:
        """使用TaskGenerator创建任务池"""
        print(f"Generating {self.num_tasks} tasks using TaskGenerator...")
        return self.task_generator.generate_tasks(self.num_tasks)

    def _precompute_normalization_factors(self):
        """预计算用于归一化的最大/最小值"""
        self.norm_factors = {}
        # 芯片属性
        self.norm_factors['max_t1'] = max(q.t1 for q in self.chip_model.values())
        self.norm_factors['max_t2'] = max(q.t2 for q in self.chip_model.values())
        self.norm_factors['min_f1q'] = min(q.fidelity_1q for q in self.chip_model.values())
        self.norm_factors['max_f1q'] = max(q.fidelity_1q for q in self.chip_model.values())
        # 任务属性
        self.norm_factors['max_task_qubits'] = max(t.num_qubits for t in self.task_pool.values())
        self.norm_factors['max_task_depth'] = max(t.depth for t in self.task_pool.values())
        self.norm_factors['max_task_duration'] = max(t.estimated_duration for t in self.task_pool.values())
        self.norm_factors['max_task_shots'] = max(t.shots for t in self.task_pool.values())

    def _normalize(self, value, min_val, max_val):
        """将值归一化到[-1, 1]范围"""
        if max_val == min_val: return 0.0
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def _get_obs(self, current_task_id: int, placement_in_progress: Dict[int, Tuple[int, int]]) -> Dict:
        """
        生成一个详细的、归一化的、可直接输入神经网络的观察。
        """
        # --- 1. 芯片状态嵌入 (Qubit Embeddings) ---
        qubit_embeddings = np.zeros(self.observation_space["qubit_embeddings"].shape, dtype=np.float32)

        # 动态获取当前调度方案的最大时间，用于归一化 next_available_time
        max_current_time = 1.0  # 避免除以0
        if self.schedule_plan:
            max_current_time = max(t['end_time'] for t in self.schedule_plan)

        for q_id, qubit in self.chip_model.items():
            idx = self.qubit_id_to_idx[q_id]

            # a. 静态自身属性
            f_1q_norm = self._normalize(qubit.fidelity_1q, self.norm_factors['min_f1q'], self.norm_factors['max_f1q'])
            f_ro_norm = self._normalize(qubit.fidelity_readout, 0.95, 1.0)  # 假设范围
            t1_norm = qubit.t1 / self.norm_factors['max_t1']
            t2_norm = qubit.t2 / self.norm_factors['max_t2']
            x_norm = qubit.id[0] / (self.chip_size[0] - 1)
            y_norm = qubit.id[1] / (self.chip_size[1] - 1)

            # b. 动态属性
            next_avail_time_norm = qubit.get_next_available_time() / max_current_time if max_current_time > 0 else 0

            qubit_features = [f_1q_norm, f_ro_norm, t1_norm, t2_norm, x_norm, y_norm, next_avail_time_norm]

            # c. 与邻居的交互属性 (固定顺序: 上, 下, 左, 右)
            for neighbor_dir in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Up, Down, Left, Right
                neighbor_id = (qubit.id[0] + neighbor_dir[0], qubit.id[1] + neighbor_dir[1])
                if neighbor_id in qubit.connectivity:
                    link = qubit.connectivity[neighbor_id]
                    f_2q_norm = self._normalize(link['fidelity_2q'], 0.95, 1.0)  # 假设范围
                    crosstalk_norm = link['crosstalk_coeff']  # 假设范围在0-1
                    qubit_features.extend([f_2q_norm, crosstalk_norm])
                else:
                    # 如果没有邻居，用0填充
                    qubit_features.extend([0.0, 0.0])

            qubit_embeddings[idx] = qubit_features

        # --- 2. 当前任务嵌入 (Current Task Embedding) ---
        task = self.task_pool[current_task_id]

        # a. 标量特征
        num_q_norm = task.num_qubits / self.norm_factors['max_task_qubits']
        depth_norm = task.depth / self.norm_factors['max_task_depth']
        duration_norm = task.estimated_duration / self.norm_factors['max_task_duration']
        shots_norm = task.shots / self.norm_factors['max_task_shots']

        # b. 交互图的统计特征
        graph = task.interaction_graph
        num_edges = graph.number_of_edges()
        degrees = [d for n, d in graph.degree()]
        mean_degree = np.mean(degrees) if degrees else 0

        # 归一化 (分母是理论最大值)
        max_edges = task.num_qubits * (task.num_qubits - 1) / 2
        num_edges_norm = num_edges / max_edges if max_edges > 0 else 0
        max_degree = task.num_qubits - 1
        mean_degree_norm = mean_degree / max_degree if max_degree > 0 else 0

        task_embedding = np.array([
            num_q_norm, depth_norm, duration_norm, shots_norm,
            num_edges_norm, mean_degree_norm
        ], dtype=np.float32)

        # --- 3. 放置掩码 (Placement Mask) ---
        placement_mask = np.zeros(self.num_qubits, dtype=np.int8)
        for physical_q_id in placement_in_progress.values():
            placement_mask[self.qubit_id_to_idx[physical_q_id]] = 1

        # --- 4. 当前逻辑比特上下文 (Logical Qubit Context) ---
        current_logical_idx = len(placement_in_progress)

        # 计算当前逻辑比特与已放置比特的连接数
        connectivity_to_placed = 0
        if current_logical_idx < task.num_qubits:
            for placed_logical_idx, placed_physical_id in placement_in_progress.items():
                if graph.has_edge(current_logical_idx, placed_logical_idx):
                    connectivity_to_placed += 1

        logical_qubit_context = np.array([
            current_logical_idx / self.max_qubits_per_task,
            connectivity_to_placed / self.max_qubits_per_task
        ], dtype=np.float32)

        return {
            "qubit_embeddings": qubit_embeddings,
            "current_task_embedding": task_embedding,
            "placement_mask": placement_mask,
            "logical_qubit_context": logical_qubit_context
        }

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """重置环境到初始状态"""
        super().reset(seed=seed)

        self.current_step = 0
        self.schedule_plan = []
        for task in self.task_pool.values():
            task.reset()
        for qubit in self.chip_model.values():
            qubit.reset()

        return self._get_obs(), {}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步调度（安排一个任务）。
        action 字典应包含: 'task_id', 'mapping', 'start_time'
        """
        task_id = action["task_id"]
        mapping = action["mapping"]  # e.g., {logic_q0: (0,1), logic_q1: (0,2)}
        start_time = action["start_time"]

        task = self.task_pool[task_id]

        # --- 验证动作 (简化版) ---
        if task.is_scheduled:
            raise ValueError(f"Task {task_id} has already been scheduled.")
        for physical_q_id in mapping.values():
            qubit = self.chip_model[physical_q_id]
            if qubit.get_next_available_time() > start_time:
                # 这是一个错误/非法的动作，实际应用中需要处理
                # 这里我们先假设智能体总是给出合法的动作
                pass

        # --- 执行动作 ---
        task.is_scheduled = True
        self.current_step += 1

        # 估算和记录
        duration = task.estimated_duration
        end_time = start_time + duration
        fidelity = self._estimate_fidelity(task, mapping)
        crosstalk = self._calculate_crosstalk(mapping, start_time, end_time)

        # 更新芯片资源预定
        for physical_q_id in mapping.values():
            self.chip_model[physical_q_id].booking_schedule.append((start_time, end_time))
            # 保持排序
            self.chip_model[physical_q_id].booking_schedule.sort()

        self.schedule_plan.append({
            "task_id": task_id,
            "mapping": mapping,
            "start_time": start_time,
            "end_time": end_time,
            "fidelity": fidelity,
            "crosstalk": crosstalk
        })

        # --- 确定奖励和终止条件 ---
        terminated = (self.current_step == self.num_tasks)
        reward = 0.0
        if terminated:
            reward = self._calculate_final_reward()

        return self._get_obs(), reward, terminated, False, {}

    def _estimate_fidelity(self, task: QuantumTask, mapping: Dict) -> float:
        # 占位符：基于所用比特的平均保真度
        fids = [self.chip_model[p_id].fidelity_1q for p_id in mapping.values()]
        # 简化模型：保真度随深度指数下降
        return np.mean(fids) ** task.depth

    def _calculate_crosstalk(self, mapping: Dict, start_time: float, end_time: float) -> float:
        # 占位符：计算与已调度任务的串扰
        score = 0.0
        for scheduled in self.schedule_plan:
            # 检查时间重叠
            if max(start_time, scheduled['start_time']) < min(end_time, scheduled['end_time']):
                # 检查物理邻接
                for p_id1 in mapping.values():
                    for p_id2 in scheduled['mapping'].values():
                        if p_id2 in self.chip_model[p_id1].connectivity:
                            score += self.chip_model[p_id1].connectivity[p_id2]['crosstalk_coeff']
        return score

    def _calculate_final_reward(self) -> float:
        """计算整个调度方案的最终奖励"""
        if not self.schedule_plan:
            return -1e6  # 惩罚空方案

        # 1. 时间奖励
        makespan = max(t['end_time'] for t in self.schedule_plan)
        # 假设一个简单的基准Makespan
        baseline_makespan = sum(t.estimated_duration for t in self.task_pool.values())
        r_time = baseline_makespan / makespan if makespan > 0 else 10  # 避免除以0

        # 2. 保真度奖励
        fidelities = [t['fidelity'] for t in self.schedule_plan]
        # 使用几何平均值
        avg_fidelity = np.prod(fidelities) ** (1 / len(fidelities)) if fidelities else 0
        r_fidelity = avg_fidelity ** 2  # 放大差距

        # 3. 串扰惩罚
        total_crosstalk = sum(t['crosstalk'] for t in self.schedule_plan)
        # 归一化（非常粗略）
        norm_crosstalk = total_crosstalk / (self.num_tasks * self.num_qubits)
        r_crosstalk = -norm_crosstalk

        # 加权求和
        w_t, w_f, w_c = 1.0, 0.5, 0.2  # 超参数，需要调整
        final_reward = w_t * r_time + w_f * r_fidelity + w_c * r_crosstalk

        return final_reward

    def render(self, mode='human'):
        """可视化调度方案"""
        if mode == 'human':
            if not self.schedule_plan:
                print("No schedule to render yet.")
                return

            print("=" * 20 + " Schedule Plan " + "=" * 20)
            # 简单的文本渲染
            sorted_plan = sorted(self.schedule_plan, key=lambda x: x['start_time'])
            for item in sorted_plan:
                print(f"Task {item['task_id']}: "
                      f"Time [{item['start_time']:.2f}, {item['end_time']:.2f}], "
                      f"Fidelity={item['fidelity']:.4f}, "
                      f"Crosstalk={item['crosstalk']:.2f}")
            makespan = max(t['end_time'] for t in self.schedule_plan)
            print(f"\nFinal Makespan: {makespan:.2f}")
            print("=" * 55)


if __name__ == '__main__':
    # --- 一个简单的使用示例 ---
    env = QuantumSchedulingEnv()
    obs, info = env.reset()

    # 模拟一个完整的Episode，使用随机但合法的动作
    for i in range(env.num_tasks):
        # 这是一个简化的、手动的智能体决策过程
        # 实际中，这是由你的Transformer模型完成的

        # 1. 选择一个未调度的任务
        task_id = -1
        for tid, task in env.task_pool.items():
            if not task.is_scheduled:
                task_id = tid
                break

        # 2. 找到一个可行的映射和时间 (非常简单的贪心策略)
        chosen_task = env.task_pool[task_id]
        mapping = {}
        start_time = 0

        # 寻找可用的比特区域
        available_qubits = [qid for qid, q in env.chip_model.items() if len(mapping) < chosen_task.num_qubits]
        # 简单的线性选择
        for i in range(chosen_task.num_qubits):
            mapping[i] = available_qubits[i]

        # 计算最早开始时间
        if mapping:
            start_time = max(env.chip_model[pid].get_next_available_time() for pid in mapping.values())

        # 3. 执行动作
        action = {"task_id": task_id, "mapping": mapping, "start_time": start_time}
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated:
            print(f"Episode finished. Final Reward: {reward}")
            break
