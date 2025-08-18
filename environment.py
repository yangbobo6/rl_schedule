import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple

from models.physical_qubit import PhysicalQubit
from models.quantum_task import QuantumTask
from task_generate import TaskGenerator


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

        # --- 【修改处】定义简化的、聚焦时间的观察空间 ---
        # 芯片嵌入维度: x, y (位置), next_avail_time (时间) -> Total 3
        D_QUBIT = 3
        # 任务标量维度: num_qubits (空间需求), estimated_duration (时间需求)
        D_TASK_SCALAR = 2
        # 我们仍然保留图的连接性信息，因为它对放置（空间布局）至关重要
        D_GRAPH_STATS = 2  # num_edges, mean_degree
        D_TASK = D_TASK_SCALAR + D_GRAPH_STATS

        self.observation_space = spaces.Dict({
            # 只保留与空间和时间最相关的特征
            "qubit_embeddings": spaces.Box(low=0.0, high=1.0, shape=(self.num_qubits, D_QUBIT), dtype=np.float32),
            "current_task_embedding": spaces.Box(low=0.0, high=1.0, shape=(D_TASK,), dtype=np.float32),
            "placement_mask": spaces.Box(low=0, high=1, shape=(self.num_qubits,), dtype=np.int8),
            "logical_qubit_context": spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
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
        """【修改处】只预计算需要的因子"""
        self.norm_factors = {}
        # 任务属性
        self.norm_factors['max_task_qubits'] = max(t.num_qubits for t in self.task_pool.values()) if self.task_pool else self.max_qubits_per_task
        self.norm_factors['max_task_duration'] = max(t.estimated_duration for t in self.task_pool.values()) if self.task_pool else 1.0

    def _normalize(self, value, min_val, max_val):
        """将值归一化到[-1, 1]范围"""
        if max_val == min_val: return 0.0
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def _get_obs(self, current_task_id: int = None, placement_in_progress: Dict[int, Tuple[int, int]] = None) -> Dict:
        """
        【已修改】生成一个简化的、聚焦于时间和空间布局的观察。
        移除了所有与保真度和串扰直接相关的特征。
        """
        # --- 1. 芯片状态嵌入 (Qubit Embeddings) ---
        qubit_embeddings = np.zeros(self.observation_space["qubit_embeddings"].shape, dtype=np.float32)

        # 动态获取当前调度方案的最大时间，用于归一化 next_available_time
        max_current_time = 1.0
        if self.schedule_plan:
            makespan_so_far = max(t['end_time'] for t in self.schedule_plan)
            if makespan_so_far > 0:
                max_current_time = makespan_so_far

        for q_id, qubit in self.chip_model.items():
            idx = self.qubit_id_to_idx[q_id]

            # a. 位置特征 (空间)
            x_norm = qubit.id[0] / (self.chip_size[0] - 1)
            y_norm = qubit.id[1] / (self.chip_size[1] - 1)

            # b. 时间特征
            next_avail_time_norm = qubit.get_next_available_time() / max_current_time

            qubit_embeddings[idx] = [x_norm, y_norm, next_avail_time_norm]

        # --- 2. 当前任务嵌入 (Current Task Embedding) ---
        task = self.task_pool[current_task_id]

        # a. 标量特征 (空间和时间需求)
        num_q_norm = task.num_qubits / self.norm_factors['max_task_qubits']
        duration_norm = task.estimated_duration / self.norm_factors['max_task_duration']

        # b. 交互图的统计特征 (空间布局约束)
        graph = task.interaction_graph
        num_edges = graph.number_of_edges()  # 边
        degrees = [d for n, d in graph.degree()]
        mean_degree = np.mean(degrees) if degrees else 0  # 平均度数（每个节点连了多少条边，取平均）

        max_edges = task.num_qubits * (task.num_qubits - 1) / 2   # 完全图最大边数
        num_edges_norm = num_edges / max_edges if max_edges > 0 else 0
        max_degree = task.num_qubits - 1   # 最大度 节点数-1
        mean_degree_norm = mean_degree / max_degree if max_degree > 0 else 0

        task_embedding = np.array([
            num_q_norm, duration_norm,
            num_edges_norm, mean_degree_norm
        ], dtype=np.float32)   # [比特数，任务持续时间，边，度]

        # --- 3. 放置掩码 (Placement Mask) 记录了当前任务中哪些逻辑比特已经映射到物理比特 构造一个长度为 num_qubits 的 0/1 向量
        placement_mask = np.zeros(self.num_qubits, dtype=np.int8)
        for physical_q_id in placement_in_progress.values():
            placement_mask[self.qubit_id_to_idx[physical_q_id]] = 1

        # --- 4. 当前逻辑比特上下文 (Logical Qubit Context) ---
        current_logical_idx = len(placement_in_progress)

        connectivity_to_placed = 0
        if current_logical_idx < task.num_qubits:
            for placed_logical_idx in placement_in_progress.keys():
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
        """
        【已修改】重置环境到初始状态，并为第一个任务生成一个合法的初始观察。
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.schedule_plan = []
        for task in self.task_pool.values():
            task.reset()
        for qubit in self.chip_model.values():
            qubit.reset()

        # --- 关键修改处 ---
        # 在reset时，我们总是准备为第一个任务（ID为0）开始调度。
        # 因此，我们可以为它生成一个初始观察。
        # 此时，还没有任何比特被放置。
        initial_task_id = 0
        initial_placement_in_progress = {}  # 空字典

        # 调用_get_obs并传递初始上下文
        initial_obs = self._get_obs(initial_task_id, initial_placement_in_progress)

        return initial_obs, {}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步宏观调度（提交一个已规划好的任务）。

        这个方法现在原子地执行一个完整的任务调度计划，更新环境状态，
        并返回一个形式上的“下一步观察”。

        Args:
            action (Dict[str, Any]): 一个包含 'task_id', 'mapping', 'start_time' 的字典。
        """
        # --- 1. 记录执行动作前的状态 ---
        makespan_before = 0
        if self.schedule_plan:
            makespan_before = max(t['end_time'] for t in self.schedule_plan)

        task_id = action["task_id"]
        mapping = action["mapping"]
        start_time = action["start_time"]

        task = self.task_pool[task_id]

        # --- 验证 (简化版) ---
        if task.is_scheduled:
            # 在实际应用中，这里应该抛出错误或处理异常
            # 因为这意味着我们的主循环逻辑有误
            print(f"Warning: Task {task_id} was already scheduled. Overwriting.")

        # --- 执行动作 ---
        task.is_scheduled = True
        self.current_step += 1

        duration = task.estimated_duration
        end_time = start_time + duration

        # 将完成的计划存入方案列表
        self.schedule_plan.append({
            "task_id": task_id,
            "mapping": mapping,
            "start_time": start_time,
            "end_time": end_time,
            # 在这个简化版中，保真度和串扰可以不计算，以节省时间
            # "fidelity": self._estimate_fidelity(task, mapping),
            # "crosstalk": self._calculate_crosstalk(mapping, start_time, end_time)
        })

        # 更新芯片资源预定
        for physical_q_id in mapping.values():
            self.chip_model[physical_q_id].booking_schedule.append((start_time, end_time))
            self.chip_model[physical_q_id].booking_schedule.sort()  # 保持排序

        # --- 3. 计算中间奖励 ---
        makespan_after = max(t['end_time'] for t in self.schedule_plan)
        makespan_increase = makespan_after - makespan_before

        # 核心奖励塑造：惩罚makespan的增加量。
        # 我们用负值，所以这是个惩罚。为了让数值更显著，可以乘以一个系数。
        intermediate_reward = -makespan_increase / 1000.0  # e.g., 除以1000将单位从ns级转为更小的数

        # --- 确定奖励和终止条件 ---
        terminated = (self.current_step == self.num_tasks)
        final_reward_bonus = 0.0
        if terminated:
            final_reward_bonus = self._calculate_final_reward()

        # 总奖励 = 中间奖励 + 最终通关奖励 (只在最后一步非0)
        total_reward = intermediate_reward + final_reward_bonus

        # --- 生成下一步观察 (Next Observation) ---
        # 这个观察在我们的主循环中不会被直接用于决策，但需要保持API的完整性。
        # 一个合理的做法是，返回下一个待调度任务的初始观察。
        if not terminated:
            # 如果还有任务，就准备下一个任务的观察
            next_task_id = self.current_step
            # 下一个任务的放置还未开始，所以 placement_in_progress 是空的
            next_obs = self._get_obs(next_task_id, {})
        else:
            # 如果所有任务都完成了，就用最后一个任务的初始观察作为占位符
            # 这里的观察内容不重要，因为 episode 已经结束
            last_task_id = self.num_tasks - 1
            next_obs = self._get_obs(last_task_id, {})

        # Gymnasium API 返回 (obs, reward, terminated, truncated, info)
        return next_obs, total_reward, terminated, False, {}

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
        """
        【已修改】只关注时间奖励。
        """
        if not self.schedule_plan:
            return -10.0

        makespan = max(t['end_time'] for t in self.schedule_plan)

        # 使用倒数形式以提供更强的非线性信号
        bonus_reward = 100.0 / makespan if makespan > 0 else 200.0

        return bonus_reward

    def render(self, mode='human'):
        """可视化当前调度方案"""
        if mode == 'human':
            print("\n=== Current Schedule Plan ===")
            print(f"Step: {self.current_step}/{self.num_tasks}")
            for i, entry in enumerate(self.schedule_plan):
                print(f"Task {entry['task_id']}: start={entry['start_time']:.2f}, end={entry['end_time']:.2f}")
                print(f"  Mapping: {entry['mapping']}")
            print("===========================\n")
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented.")
