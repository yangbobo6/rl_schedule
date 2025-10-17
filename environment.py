import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Tuple

from models.physical_qubit import PhysicalQubit
from models.quantum_task import QuantumTask
from task_generate import TaskGenerator
from chip_visualizer import QuantumChipVisualizer


class QuantumSchedulingEnv(gym.Env):
    """
    用于量子电路离线调度的Gymnasium环境。
    智能体的任务是为一系列量子任务生成一个完整的调度方案。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, chip_size: Tuple[int, int], num_tasks: int,
                 max_qubits_per_task: int, gnn_model, device, gate_times: Dict[str, float], reward_weights: Dict[str, float]
                 ):
        super().__init__()

        self.chip_size = chip_size
        self.num_qubits = chip_size[0] * chip_size[1]
        self.num_tasks = num_tasks
        self.max_qubits_per_task = max_qubits_per_task  # 用于定义观察空间
        self.reward_weights = reward_weights

        # --- GNN编码器 (用于将任务交互图编码为向量) ---
        # 在实际项目中，这会是一个预训练或端到端训练的PyTorch/TensorFlow模型
        # 这里我们用一个虚拟的实现代替
        # self.gnn_encoder = GNNEncoder(embedding_dim=16

        # --- 核心数据结构 ---
        self.task_generator = TaskGenerator(gate_times, gnn_model, device)
        self.chip_model: Dict[Tuple[int, int], PhysicalQubit] = self._create_chip_model()
        self.task_pool: Dict[int, QuantumTask] = self._create_task_pool()
        self.schedule_plan: List[Dict] = []
        self.current_step = 0
        # 简化版SWAP估算器：假设每个SWAP门增加固定的时间和错误
        self.swap_penalty = {"duration": 3 * 350, "error": 0.03}  # 3个CX门
        
        # --- 可视化工具 ---
        self.visualizer = QuantumChipVisualizer(self.chip_model, self.chip_size)

        # 观察空间的定义现在依赖于gnn_model的输出维度
        gnn_output_dim = gnn_model.conv2.out_channels

        # --- 预计算用于归一化的值 ---
        self._precompute_normalization_factors()

        self.sorted_physical_qubits = self._rank_physical_qubits()

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

        # --- 观察空间需要包含所有物理细节 ---
        # 芯片嵌入维度: x, y, next_avail_time (3)
        #             + f_1q, f_ro (2)
        #             + 4 * (f_2q) for neighbors (4) -> Total 9
        # (暂时不在状态中直接编码串扰，而是通过奖励让模型隐式学习)
        D_QUBIT = 9
        # 任务标量维度: num_qubits (空间需求), estimated_duration (时间需求)
        D_TASK_SCALAR = 2
        # 我们仍然保留图的连接性信息，因为它对放置（空间布局）至关重要
        D_GRAPH_STATS = 2  # num_edges, mean_degree

        D_GRAPH_EMBED = gnn_output_dim
        D_TASK = D_TASK_SCALAR + D_GRAPH_EMBED

        # 1 (index) + 1 (connectivity_count) + 2(position) = 4
        D_LOGICAL_CONTEXT = 4

        print(f"Env Init: D_QUBIT={D_QUBIT}, D_TASK={D_TASK}")

        self.observation_space = spaces.Dict({
            # 只保留与空间和时间最相关的特征
            "qubit_embeddings": spaces.Box(low=0.0, high=1.0, shape=(self.num_qubits, D_QUBIT), dtype=np.float32),
            "task_embeddings": spaces.Box(low=-1.0, high=1.0, shape=(self.num_tasks, D_TASK), dtype=np.float32),
            "placement_mask": spaces.Box(low=0, high=1, shape=(self.num_qubits,), dtype=np.int8),
            "logical_qubit_context": spaces.Box(low=0, high=self.max_qubits_per_task, shape=(D_LOGICAL_CONTEXT,), dtype=np.float32)
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
        """使用TaskGenerator从large_task_pool中采样创建任务池"""
        print(f"Sampling {self.num_tasks} tasks from TaskGenerator's large_task_pool...")
        return self.task_generator.get_episode_tasks(self.num_tasks)

    def _rank_physical_qubits(self) -> list:
        """
        辅助函数：计算每个物理比特的综合质量得分，并按得分从高到低排序。
        """
        qubit_scores = []
        for q_id, qubit in self.chip_model.items():
            # 计算平均2Q门保真度
            avg_f2q = 0.0
            if qubit.connectivity:
                avg_f2q = np.mean([link['fidelity_2q'] for link in qubit.connectivity.values()])

            # 质量分 = w1 * 度数 + w2 * 平均2Q保真度 + w3 * 1Q保真度
            # 这里的权重可以根据经验调整
            score = (
                    0.2 * len(qubit.connectivity) +  # 度数权重
                    0.5 * (avg_f2q - 0.95) +  # 2Q保真度权重 (减去基准值)
                    0.3 * (qubit.fidelity_1q - 0.99)  # 1Q保真度权重
            )
            qubit_scores.append((qubit, score))

        # 按得分降序排序
        sorted_qubits = [q for q, score in sorted(qubit_scores, key=lambda x: x[1], reverse=True)]
        return sorted_qubits

    def _precompute_normalization_factors(self):
        """只预计算需要的因子"""
        self.norm_factors = {}
        # 任务属性
        self.norm_factors['max_task_qubits'] = max(
            t.num_qubits for t in self.task_pool.values()) if self.task_pool else self.max_qubits_per_task
        self.norm_factors['max_task_duration'] = max(
            t.estimated_duration for t in self.task_pool.values()) if self.task_pool else 1.0

        self.norm_factors['min_f1q'] = min(q.fidelity_1q for q in self.chip_model.values())
        self.norm_factors['max_f1q'] = max(q.fidelity_1q for q in self.chip_model.values())

        self.norm_factors['min_fro'] = 0.95
        self.norm_factors['max_fro'] = 1.0
        self.norm_factors['min_f2q'] = 0.95
        self.norm_factors['max_f2q'] = 1.0

    def _normalize(self, value, min_val, max_val):
        """将值归一化到[-1, 1]范围"""
        if max_val == min_val: return 0.0
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def _get_obs(self,
                 placement_in_progress: Dict[int, Tuple[int, int]] = None,
                 task_for_placement: QuantumTask = None) -> Dict:
        """
        生成一个简化的、聚焦于时间和空间布局的观察。
        确保 task_embedding 的构建与 observation_space 的定义一致。
        """
        # --- 1. 芯片状态嵌入 (Qubit Embeddings) ---
        qubit_embeddings = np.zeros(self.observation_space["qubit_embeddings"].shape, dtype=np.float32)

        max_current_time = 1.0  # 初始化防止归一化时间除0错误
        if self.schedule_plan:
            makespan_so_far = max(t['end_time'] for t in self.schedule_plan)
            if makespan_so_far > 0:
                max_current_time = makespan_so_far

        for q_id, qubit in self.chip_model.items():
            # a. 空间 & 时间
            idx = self.qubit_id_to_idx[q_id]
            x_norm = qubit.id[0] / (self.chip_size[0] - 1)
            y_norm = qubit.id[1] / (self.chip_size[1] - 1)
            next_avail_time_norm = qubit.get_next_available_time() / max_current_time

            # b. 保真度特征  单量子比特门和读出保真度
            f_1q_norm = self._normalize(qubit.fidelity_1q, self.norm_factors['min_f1q'], self.norm_factors['max_f1q'])
            f_ro_norm = self._normalize(qubit.fidelity_readout, self.norm_factors['min_fro'], self.norm_factors['max_fro'])  # 假设范围
            qubit_features = [x_norm, y_norm, next_avail_time_norm, f_1q_norm, f_ro_norm]

            # c. 邻居2Q门保真度 (用于学习连通性)，上下左右将其加到后面
            for neighbor_dir in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor_id = (qubit.id[0] + neighbor_dir[0], qubit.id[1] + neighbor_dir[1])
                if neighbor_id in qubit.connectivity:
                    link = qubit.connectivity[neighbor_id]
                    f_2q_norm = self._normalize(link['fidelity_2q'], self.norm_factors['min_f2q'], self.norm_factors['max_f2q'])
                    qubit_features.append(f_2q_norm)
                else:
                    qubit_features.append(0.0)  # 用0填充

            qubit_embeddings[idx] = qubit_features

        # --- 2. 任务池嵌入和掩码 (总是计算) ---
        # 即使在放置阶段，我们也计算所有任务的信息，Transformer模型可以学会只关注需要的信息。
        task_embeddings = np.zeros(self.observation_space["task_embeddings"].shape, dtype=np.float32)
        task_mask = np.ones(self.num_tasks, dtype=np.int8)  # 1 = 非法/已调度

        for task_idx, task in self.task_pool.items():
            # a. 标量特征
            num_q_norm = task.num_qubits / self.norm_factors['max_task_qubits']
            duration_norm = task.estimated_duration / self.norm_factors['max_task_duration']
            scalar_features = np.array([num_q_norm, duration_norm], dtype=np.float32)

            # b. GNN嵌入
            graph_embedding = task.graph_embedding

            # c. 拼接
            task_embedding_vector = np.concatenate([scalar_features, graph_embedding])

            # 填充到总的任务嵌入张量中
            task_embeddings[task_idx] = task_embedding_vector

            # 如果任务未被调度，则在掩码中标记为合法 (0)
            if not task.is_scheduled:
                task_mask[task_idx] = 0

        # --- 3. 放置掩码 (Placement Mask) ---
        placement_mask = np.zeros(self.num_qubits, dtype=np.int8)
        if placement_in_progress:
            for physical_q_id in placement_in_progress.values():
                placement_mask[self.qubit_id_to_idx[physical_q_id]] = 1

        # --- 4. 当前逻辑比特上下文 (Logical Qubit Context) ---  placement_in_progress -> {0:(0,5) 1:(1,4)}
        # current_logical_idx 当前要放置的逻辑比特索引
        # 默认值为0
        current_logical_idx = 0
        connectivity_to_placed = 0
        norm_avg_x = 0.0
        norm_avg_y = 0.0

        if placement_in_progress is not None and task_for_placement is not None:
            # 只有在放置阶段，才计算这些上下文信息

            task = task_for_placement  # 使用传入的 task 对象
            current_logical_idx = len(placement_in_progress)

            placed_neighbors_positions_list = []
            if current_logical_idx < task.num_qubits:
                graph = task.interaction_graph
                for placed_logical_idx, placed_physical_id in placement_in_progress.items():
                    if graph.has_edge(current_logical_idx, placed_logical_idx):
                        connectivity_to_placed += 1
                        placed_neighbors_positions_list.append(placed_physical_id)

            if connectivity_to_placed > 0:
                avg_neighbor_x = np.mean([pos[0] for pos in placed_neighbors_positions_list])
                avg_neighbor_y = np.mean([pos[1] for pos in placed_neighbors_positions_list])
                norm_avg_x = avg_neighbor_x / (self.chip_size[0] - 1) if self.chip_size[0] > 1 else 0
                norm_avg_y = avg_neighbor_y / (self.chip_size[1] - 1) if self.chip_size[1] > 1 else 0
        # 构建最终的上下文向量
        logical_qubit_context = np.array([
            current_logical_idx / self.max_qubits_per_task,
            connectivity_to_placed / self.max_qubits_per_task,
            norm_avg_x,
            norm_avg_y
        ], dtype=np.float32)

        return {
            "qubit_embeddings": qubit_embeddings,
            "task_embeddings": task_embeddings,
            "task_mask": task_mask,
            "placement_mask": placement_mask,
            "logical_qubit_context": logical_qubit_context
        }

    def _estimate_swaps_and_fidelity(self, task: QuantumTask, mapping: Dict) -> Tuple[int, float]:
        """
        辅助函数：估算给定映射所需的SWAP数量和最终保真度。
        """
        num_swaps = 0
        graph = task.interaction_graph

        # 遍历任务交互图中的每一条边（代表一个2Q门需求）
        for u, v in graph.edges():
            p_u, p_v = mapping[u], mapping[v]
            # 如果物理上不相邻，就需要SWAP
            if p_v not in self.chip_model[p_u].connectivity:
                # 这是一个简化的估算，实际的SWAP数量取决于路由算法
                # 我们这里简单地认为每条不满足的边都需要一次SWAP
                num_swaps += 1

        # 估算保真度
        # 1. 基础保真度，来自所选比特
        avg_f1q = np.mean([self.chip_model[pid].fidelity_1q for pid in mapping.values()])
        # 2. 考虑门操作和SWAP引入的错误
        total_error = task.depth * (1 - avg_f1q) + num_swaps * self.swap_penalty["error"]
        final_fidelity = np.exp(-total_error)  # 一个常用的近似模型

        return num_swaps, final_fidelity

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """
        重置环境到初始状态，并为第一个任务生成一个合法的初始观察。
        """
        super().reset(seed=seed)

        self.task_pool = self.task_generator.get_episode_tasks(self.num_tasks)

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
        initial_obs = self._get_obs(placement_in_progress=None, task_for_placement=None)

        return initial_obs, {}

    def _calculate_idle_volume(self, schedule: list, makespan: float) -> float:
        """辅助函数，计算当前调度方案的空闲时空体积"""
        if not schedule:
            return 0.0

        occupied_volume = sum(
            len(item['mapping']) * (item['end_time'] - item['start_time'])
            for item in schedule
        )
        total_volume = self.num_qubits * makespan
        return total_volume - occupied_volume

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        使用“紧凑度”奖励函数。
        """
        # --- 1. 记录执行动作前的状态 ---
        makespan_before = 0
        if self.schedule_plan:
            makespan_before = max(t['end_time'] for t in self.schedule_plan)
        idle_volume_before = self._calculate_idle_volume(self.schedule_plan, makespan_before)

        # --- 2. 正常执行动作 ---
        task_id = action["task_id"]
        task = self.task_pool[task_id]
        start_time = action["start_time"]
        mapping = action["mapping"]

        # --- 在执行前，估算SWAP和保真度 ---
        num_swaps, final_fidelity = self._estimate_swaps_and_fidelity(task, mapping)

        # 因SWAP而增加的额外时长
        duration_penalty_swap = num_swaps * self.swap_penalty["duration"] / 1e3  # us

        task.is_scheduled = True
        self.current_step += 1
        duration = task.estimated_duration + duration_penalty_swap  # 总时长=估算+包含SWAP
        end_time = start_time + duration

        # 创建一个临时的、更新后的调度计划来计算新状态
        next_schedule_plan = self.schedule_plan + [{
            "task_id": task_id, "mapping": action["mapping"],
            "start_time": action["start_time"], "end_time": end_time
        }]

        # 2.5. 对当前任务的逻辑比特按度数从高到低排序
        graph = task.interaction_graph
        # graph.degree返回的是 (node, degree) 的元组列表
        sorted_logical_qubits = sorted(graph.degree, key=lambda x: x[1], reverse=True)

        match_score = 0
        num_logical_to_check = len(sorted_logical_qubits)

        # 获取芯片质量最高的前N个物理比特，N=任务大小
        top_physical_qubits = self.sorted_physical_qubits[:num_logical_to_check]

        # 遍历排序后的逻辑比特
        for i in range(num_logical_to_check):
            logical_qubit_id, _ = sorted_logical_qubits[i]

            # 找到它被映射到了哪个物理比特
            mapped_physical_id = mapping[logical_qubit_id]
            mapped_physical_qubit_obj = self.chip_model[mapped_physical_id]

            # 检查这个物理比特是否在“优等生”名单里
            # 为了奖励更精确，我们检查排名是否匹配
            # 例如，最重要的逻辑比特是否映射到了最重要的物理比特之一？

            # 简化版奖励：只要映射到了Top-N就算匹配
            if mapped_physical_qubit_obj in top_physical_qubits:
                # 权重可以根据排名递减，最重要的匹配得分最高
                # 例如，排名第一的逻辑比特，如果映射到了排名前1/3的物理比特，得分
                rank_score = (num_logical_to_check - i) / num_logical_to_check
                match_score += rank_score

        # 归一化匹配分数
        # 最大可能的match_score是所有rank_score的总和
        max_possible_score = sum((num_logical_to_check - i) / num_logical_to_check for i in range(num_logical_to_check))
        reward_matching = match_score / max_possible_score if max_possible_score > 0 else 0.0

        # --- 3.计算多目标的中间奖励 ---
        #  1>. 计算紧凑度奖励
        makespan_after = max(t['end_time'] for t in next_schedule_plan)
        idle_volume_after = self._calculate_idle_volume(next_schedule_plan, makespan_after)
        task_volume = len(action["mapping"]) * duration
        # 净空闲体积减少量 = (旧空闲) - (新空闲)
        idle_volume_reduction = idle_volume_before - idle_volume_after
        # 核心奖励公式：紧凑度得分
        # 这个得分衡量了每单位任务体积能带来多大的净空闲体积减少
        # 它是无界的，但可以用tanh将其映射到一个好的范围[-1, 1]
        compaction_score = (idle_volume_reduction / task_volume) if task_volume > 0 else 0.0
        # 使用tanh函数将得分映射到(-1, 1)，这是一个很好的归一化技巧
        reward_compaction = np.tanh(compaction_score)

        # 2>. SWAP惩罚 (SWAP Penalty)
        # 惩罚与SWAP数量的对数成正比，避免数值过大
        swap_duration_penalty = num_swaps * self.swap_penalty["duration"] / 1e3  # us
        # a. 基于数量的惩罚 (线性)
        penalty_swap_count = -num_swaps
        # b. 基于时间的惩罚 (将时间惩罚也纳入奖励)
        penalty_swap_time = -swap_duration_penalty / 1000.0  # 归一化
        # 将两者结合
        penalty_swap = penalty_swap_count + penalty_swap_time

        # 3>. 保真度奖励 (Fidelity Reward)
        # 保真度本身就在[0,1]区间，是一个很好的奖励
        min_acceptable_fidelity = 0.80
        reward_fidelity = (final_fidelity - min_acceptable_fidelity) / (1.0 - min_acceptable_fidelity)
        reward_fidelity = np.clip(reward_fidelity, 0, 1.0)  # 裁剪到[0, 1]

        # 4>. 串扰惩罚 (Crosstalk Penalty)   mapping
        crosstalk_score = self._calculate_crosstalk(mapping, start_time, end_time)
        # 归一化并取负值
        penalty_crosstalk = -crosstalk_score * 5.0   # 粗略归一化

        # --- 加权求和得到总中间奖励 ---
        w = self.reward_weights  # 关键超参数！

        intermediate_reward = (w["compaction"] * reward_compaction +
                               w["swap"] * penalty_swap +
                               w["fidelity"] * reward_fidelity +
                               w["crosstalk"] * penalty_crosstalk +
                               w["matching"] * reward_matching)  # <-- 新增奖励项

        # 5>. 确定终止和最终奖励 ---
        terminated = (self.current_step == self.num_tasks)
        final_reward_bonus = 0.0
        if terminated:
            # 最终奖励保持不变，作为对最终makespan的评价
            final_reward_bonus = self._calculate_final_reward()

        total_reward = intermediate_reward + final_reward_bonus

        info = {
            "num_swaps": num_swaps,
            "final_fidelity": final_fidelity,
            "crosstalk_score": crosstalk_score
        }

        # --- 4. 更新真实的环境状态 ---
        self.schedule_plan = next_schedule_plan
        for physical_q_id in action["mapping"].values():
            self.chip_model[physical_q_id].booking_schedule.append((action["start_time"], end_time))
            self.chip_model[physical_q_id].booking_schedule.sort()

        # ... 生成下一步观察并返回...
        if not terminated:
            next_obs = self._get_obs(placement_in_progress=None, task_for_placement=None)
        else:
            next_obs = self._get_obs(placement_in_progress=None, task_for_placement=None)

        return next_obs, total_reward, terminated, False, info

    def _estimate_fidelity(self, task: QuantumTask, mapping: Dict) -> float:
        # 占位符：基于所用比特的平均保真度
        fids = [self.chip_model[p_id].fidelity_1q for p_id in mapping.values()]
        # 简化模型：保真度随深度指数下降
        return np.mean(fids) ** task.depth

    def _calculate_crosstalk(self, mapping: Dict, start_time: float, end_time: float) -> float:
        # 占位符：计算与已调度任务的串扰
        score = 0.0
        for scheduled in self.schedule_plan:
            # 检查时间重叠  例如：mapping {0: (3, 4), 1: (3, 3), 2: (4, 3), 3: (3, 2)}
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
    
    def visualize_chip(self, save_path: str = None, show_values: bool = True):
        """可视化量子芯片的物理属性和连接性"""
        self.visualizer.visualize_chip_overview(save_path, show_values)
    
    def visualize_scheduling_state(self, save_path: str = None):
        """可视化当前调度状态"""
        current_time = 0
        if self.schedule_plan:
            current_time = max(t['end_time'] for t in self.schedule_plan)
        self.visualizer.visualize_scheduling_state(self.schedule_plan, current_time, save_path)
    
    def visualize_connectivity(self, save_path: str = None):
        """可视化芯片连接矩阵"""
        self.visualizer.visualize_connectivity_with_edge_labels(save_path)
    
    def export_chip_stats(self, save_path: str = None):
        """导出芯片统计信息"""
        return self.visualizer.export_chip_stats(save_path)
