# 文件: models/hsamesq_scheduler.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import math


class HSAMeSQScheduler:
    """
    H-SAMeS-Q: Hierarchical Shot-Aware Measurement Scheduler
               with Quota-based Adaptive Micro-batching

    分层 shot 感知 + 测量同步合批 + 配额驱动自适应微批调度器。
    """

    def __init__(self, env, config=None):
        self.env = env

        # 默认参数 (可从 config 覆盖)
        self.config = config or {}
        self.B_MIN = self.config.get('B_MIN', 10)  # 最小微批
        self.B_MAX = self.config.get('B_MAX', 50)  # 最大微批
        self.DELTA = self.config.get('DELTA', 0.5)  # 同步等待阈值 (us)

        # 权重参数
        self.W_S = self.config.get('W_S', 1.0)  # 剩余shot权重
        self.W_D = self.config.get('W_D', 0.5)  # 时长权重
        self.W_DEG = self.config.get('W_DEG', 0.2)  # 冲突度权重
        self.W_AGE = self.config.get('W_AGE', 2.0)  # 等待时间权重 (防饥饿)

    def _precompute_compatibility(self, tasks):
        """
        预计算兼容性矩阵。
        Compatible[i][j] = 1 if (Qubits Disjoint AND (Readout Disjoint OR Readout Subset))
        """
        num_tasks = len(tasks)
        compatibility_matrix = np.zeros((num_tasks, num_tasks), dtype=bool)

        # 提取集合以加速
        task_lines = [set(t['readout_lines']) for t in tasks]
        task_qubits = [t['occupied_qubits'] for t in tasks]  # 新增

        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    compatibility_matrix[i][j] = False
                    continue

                # --- 1. 检查物理比特冲突 (最高优先级) ---
                qubits_i = task_qubits[i]
                qubits_j = task_qubits[j]

                if not qubits_i.isdisjoint(qubits_j):
                    # 如果有公共比特，绝对不兼容！
                    compatibility_matrix[i][j] = False
                    continue

                    # --- 2. 检查测量线冲突 ---
                lines_i = task_lines[i]
                lines_j = task_lines[j]

                # 规则：不冲突 OR 包含关系
                is_readout_disjoint = lines_i.isdisjoint(lines_j)
                is_subset = lines_i.issubset(lines_j) or lines_j.issubset(lines_i)

                if is_readout_disjoint or is_subset:
                    compatibility_matrix[i][j] = True
                else:
                    compatibility_matrix[i][j] = False

        return compatibility_matrix, task_lines

    def _calculate_batch_size(self, task, contention_score):
        """
        计算自适应微批大小 B_i
        B_i = clip(B_min + k1 * age - k2 * contention, B_min, B_max)
        """
        # 简单线性公式，参数可调
        k1 = 0.5
        k2 = 10.0

        base_b = self.B_MIN + k1 * task['age'] - k2 * contention_score

        # 确保不超过剩余 shot 数
        b_i = int(np.clip(base_b, self.B_MIN, self.B_MAX))
        return min(b_i, task['remaining_shots'])

    def schedule(self, initial_plan: list, shots_per_task: int = 100) -> dict:
        print(f"\n--- Running H-SAMeS-Q Scheduler (Shots={shots_per_task}) ---")

        # 1. 初始化任务状态
        tasks = []
        for i, item in enumerate(initial_plan):
            # 获取该任务占用的物理比特集合
            # item['mapping'] 是 {logic_id: phys_id}，我们取 values
            occupied_qubits = set(item['mapping'].values())
            task_id = item['task_id']
            # 物理持续时间
            duration = item['end_time'] - item['start_time']
            # 测量线
            readout_lines = self.env.get_task_readout_lines(item['mapping'])

            tasks.append({
                'index': i,  # 内部索引
                'id': task_id,
                'duration': duration,
                'readout_lines': list(readout_lines),  # 转列表方便后续处理
                'remaining_shots': shots_per_task,
                'age': 0,  # 等待轮次
                'shots_schedule': [],
                'occupied_qubits': occupied_qubits,
            })

        # 2. 预计算
        compatibility_matrix, task_line_sets = self._precompute_compatibility(tasks)

        # 计算静态冲突度 (Degree in Conflict Graph)
        # 冲突 = 不兼容
        conflict_degrees = []
        for i in range(len(tasks)):
            # 兼容矩阵取反就是冲突矩阵 (对角线除外)
            num_compatible = np.sum(compatibility_matrix[i])
            # conflict_degree = (N-1) - num_compatible
            # normalized to [0, 1]
            conflict_degree = 1.0 - (num_compatible / (len(tasks) - 1)) if len(tasks) > 1 else 0
            conflict_degrees.append(conflict_degree)

        # 3. 主循环
        current_time = 0.0
        total_shots = len(tasks) * shots_per_task
        pbar = tqdm(total=total_shots, desc="H-SAMeS-Q Scheduling")

        while True:
            # a. 筛选候选任务 (还有剩余shot)
            active_indices = [i for i, t in enumerate(tasks) if t['remaining_shots'] > 0]
            if not active_indices:
                break

            # b. 计算优先级权重 w_i
            # w_i = a1*S + a2*d + a3*deg + a4*age
            weights = []
            for idx in active_indices:
                t = tasks[idx]
                w = (self.W_S * t['remaining_shots'] +
                     self.W_D * t['duration'] +
                     self.W_DEG * conflict_degrees[idx] +
                     self.W_AGE * t['age'])
                weights.append((w, idx))

            # 按权重降序排列
            weights.sort(key=lambda x: x[0], reverse=True)
            sorted_indices = [x[1] for x in weights]

            # c. 构建本轮任务集合 R_k (贪心最大兼容集)
            round_tasks_indices = []

            for idx in sorted_indices:
                # 检查与已选任务的兼容性
                is_compatible = True
                for selected_idx in round_tasks_indices:
                    if not compatibility_matrix[idx][selected_idx]:
                        is_compatible = False
                        break

                if is_compatible:
                    round_tasks_indices.append(idx)

            # d. 确定微批大小 B_i
            micro_batch_configs = []  # List of (task_idx, batch_size)

            for idx in round_tasks_indices:
                b_i = self._calculate_batch_size(tasks[idx], conflict_degrees[idx])
                micro_batch_configs.append((idx, b_i))

            # e. 计算本轮时间参数 (对齐与同步)
            # D_k = max(B_i * d_i)
            # 在这里，我们考虑同步阈值 DELTA
            # 如果 B_i * d_i 与最大值相差在 DELTA 以内，就对齐
            # 否则... (在这个简化版中，我们总是强制对齐到末端，这是最鲁棒的)

            round_durations = [b * tasks[idx]['duration'] for idx, b in micro_batch_configs]
            round_max_duration = max(round_durations)
            round_end_time = current_time + round_max_duration

            # f. 安排并更新
            for (idx, b_i), my_duration in zip(micro_batch_configs, round_durations):
                t = tasks[idx]

                # 计算对齐产生的等待 (slack)
                slack = round_max_duration - my_duration

                # 如果 slack > DELTA，理论上可以选择不完全对齐（提前结束），
                # 但为了保证下一轮的干净开始，通常还是对齐。
                # 这里的优化空间在于：是否可以通过调整 b_i 来减少 slack？
                # (这是更高阶的优化，暂不实现，保持简单)

                # 记录调度结果 (start, end)
                # 对齐到末端：start = round_end_time - my_duration
                actual_start = round_end_time - my_duration
                actual_end = round_end_time

                # 记录 b_i 次 shot
                # 在宏观甘特图上，这表现为一个长块
                # 我们只记录这个微批的起止，而不是微批内每次shot的起止
                t['shots_schedule'].append({
                    'start': actual_start,
                    'end': actual_end,
                    'num_shots': b_i,
                    'batch_id': len(t['shots_schedule'])
                })

                # 更新状态
                t['remaining_shots'] -= b_i
                t['age'] = 0  # 重置年龄，因为它被服务了
                pbar.update(b_i)

            # g. 更新未被选中任务的年龄
            selected_set = set(round_tasks_indices)
            for idx in active_indices:
                if idx not in selected_set:
                    tasks[idx]['age'] += 1

            # h. 推进时间
            current_time = round_end_time

        pbar.close()

        return {
            "makespan": current_time,
            "tasks": tasks
        }