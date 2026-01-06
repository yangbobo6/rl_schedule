# 文件: models/hsamesq_scheduler.py

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import math


class HSAMeSQSchedulerV2:
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
        self.W_AGE = self.config.get('W_AGE', 5.0)  # 等待时间权重 (防饥饿)


    def _identify_critical_tasks(self, tasks, shots_per_task=None):
        # 注意：shots_per_task 参数现在没用了，因为 tasks 里的 item 已经有了 remaining_shots
        """
        【v3 修改】识别关键路径任务。
        """
        qubit_load = defaultdict(float)
        task_usage = defaultdict(list)

        for idx, task in enumerate(tasks):
            # 使用任务自己的 shot 数
            total_duration = task['duration'] * task['remaining_shots']

            for qid in task['occupied_qubits']:
                qubit_load[qid] += total_duration
                task_usage[qid].append(idx)

        if not qubit_load: return [], set()

        critical_qubit = max(qubit_load, key=qubit_load.get)
        critical_task_indices = set(task_usage[critical_qubit])

        print(f"  [Critical Path] Qubit {critical_qubit} is the bottleneck.")
        print(f"  [Critical Path] Load: {qubit_load[critical_qubit]:.2f} us")

        return list(critical_task_indices), critical_task_indices


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

                compatibility_matrix[i][j] = True

                    # --- 2. 检查测量线冲突 ---
                # lines_i = task_lines[i]
                # lines_j = task_lines[j]
                #
                # # 规则：不冲突 OR 包含关系
                # is_readout_disjoint = lines_i.isdisjoint(lines_j)
                # is_subset = lines_i.issubset(lines_j) or lines_j.issubset(lines_i)
                #
                # if is_readout_disjoint or is_subset:
                #     compatibility_matrix[i][j] = True
                # else:
                #     compatibility_matrix[i][j] = False

        return compatibility_matrix, task_lines


    def _calculate_batch_size(self, task, contention_score, is_critical):
        """
        计算自适应微批大小 B_i。
        关键任务倾向于更大的批次以减少切换开销。
        """
        k1 = 0.5
        k2 = 10.0

        # 关键任务获得额外的批次增益
        critical_boost = 10 if is_critical else 0

        base_b = self.B_MIN + critical_boost + k1 * task['age'] - k2 * contention_score

        b_i = int(np.clip(base_b, self.B_MIN, self.B_MAX))
        return min(b_i, task['remaining_shots'])

    def schedule(self, initial_plan: list, shots_per_task: int = 100) -> dict:
        print(f"\n--- Running H-SAMeS-Q v3 (Run-to-Completion Strategy) ---")

        # 1. 初始化任务状态 (保持不变)
        tasks = []
        for i, item in enumerate(initial_plan):
            occupied_qubits = set(item['mapping'].values())
            task_id = item['task_id']
            duration = item['end_time'] - item['start_time']
            readout_lines = self.env.get_task_readout_lines(item['mapping'])

            # 获取任务对象，读取真实 shot 数
            task_obj = self.env.task_pool[task_id]

            # 如果外部传入了 shots_per_task，则覆盖（用于对比实验），否则用真实的
            real_shots = shots_per_task if shots_per_task is not None else task_obj.shots

            tasks.append({
                'index': i,
                'id': task_id,
                'duration': duration,
                'readout_lines': list(readout_lines),
                'remaining_shots': real_shots,  # 使用真实 shot 数
                'initial_shots': real_shots,  # 记录初始值，用于计算权重
                'age': 0,
                'shots_schedule': [],
                'occupied_qubits': occupied_qubits,
                # 我们可以加入 release_time，虽然目前都是0，但为未来扩展留口
                'release_time': 0
            })

            print(f"任务：{i},shots are {real_shots}")

        # 2. 预计算兼容性 (保持不变)
        compatibility_matrix, _ = self._precompute_compatibility(tasks)

        # 3. 识别关键任务 (保持不变)
        _, critical_task_indices_set = self._identify_critical_tasks(tasks)

        for i, item in enumerate(initial_plan):
            tasks[i]['mapping'] = item['mapping']

        # 4. 主循环
        current_time = 0.0
        total_shots = sum(t['remaining_shots'] for t in tasks)
        pbar = tqdm(total=total_shots, desc="H-SAMeS-Q Scheduling")

        while True:
            # a. 筛选候选任务
            active_indices = [i for i, t in enumerate(tasks) if t['remaining_shots'] > 0]
            if not active_indices:
                break

            # b. 构建本轮集合 (Run-to-Completion 策略)
            round_tasks_indices = []

            # 分离候选
            active_critical = [i for i in active_indices if i in critical_task_indices_set]
            active_non_critical = [i for i in active_indices if i not in critical_task_indices_set]

            # --- 策略修改点 1: 种子选择 ---
            # 优先从 critical 中选，如果空则从 non-critical 中选
            seed_candidates = active_critical if active_critical else active_non_critical

            if not seed_candidates: break  # Should not happen

            seed_task_idx = min(seed_candidates, key=lambda i: tasks[i]['remaining_shots'])
            round_tasks_indices.append(seed_task_idx)

            seed_duration = tasks[seed_task_idx]['duration']

            # 2. 动态排序填充
            # 我们不再一次性排好序，而是每一轮都动态选择“最合适”的下一个

            # 候选池
            candidates_pool = [i for i in active_indices if i != seed_task_idx]

            # 我们要不断从 pool 中选人，直到塞满或者没得选
            # 但为了效率，我们还是排一次序，然后遍历

            def get_multi_obj_priority(idx):
                t = tasks[idx]
                is_crit = idx in critical_task_indices_set

                # 1. SRPT 基础分 (尽快完成)
                score = 10000.0 if is_crit else 0.0
                score -= t['remaining_shots'] * t['duration']

                # 2. 【核心优化】时长匹配分
                # 差异越小，分数越高
                duration_diff = abs(t['duration'] - seed_duration)
                # 使用高斯核或者倒数来评分
                score -= 500.0 * (duration_diff / seed_duration)  # 归一化差异，权重可调

                # 3. 防饥饿 (Age)
                score += 10.0 * t['age']

                return score

            candidates_pool.sort(key=get_multi_obj_priority, reverse=True)

            for idx in candidates_pool:
                t = tasks[idx]

                # --- 检查 1: 硬约束兼容性 ---
                is_compatible = True
                for selected_idx in round_tasks_indices:
                    if not compatibility_matrix[idx][selected_idx]:
                        is_compatible = False
                        break
                if not is_compatible: continue

                # --- 检查 2: 串扰准入 ---
                # 计算如果加入这个任务，会增加多少串扰
                marginal_crosstalk = 0
                mapping_new = t['mapping']

                for selected_idx in round_tasks_indices:
                    mapping_existing = tasks[selected_idx]['mapping']
                    # 计算两组映射之间的物理邻接串扰
                    for p1 in mapping_new.values():
                        for p2 in mapping_existing.values():
                            if p2 in self.env.chip_model[p1].connectivity:
                                marginal_crosstalk += self.env.chip_model[p1].connectivity[p2]['crosstalk_coeff']

                # 阈值判断：如果串扰太高，为了保真度，拒绝加入！
                # 这个阈值可以是固定的 (e.g. 0.2)，也可以是动态的
                if marginal_crosstalk > 0.15:  # 举例阈值
                    continue

                    # --- 检查 3: 时长差异硬截断 (可选) ---
                # 如果差异实在太大 (比如 10倍)，即使排名靠前也不要
                if abs(t['duration'] - seed_duration) > 20000:  # e.g. 20us
                    continue

                # 通过所有检查，加入本轮
                round_tasks_indices.append(idx)

            # c. 确定本轮 Shot 数 (Batch Size)
            # 为了配合 "尽快做完" 的策略，我们希望 batch_size 尽可能大
            # 但要受限于本轮中 shot 最少的那个任务（木桶效应），以保证同步

            min_remaining = min(tasks[i]['remaining_shots'] for i in round_tasks_indices)

            # 直接取 min(min_remaining, B_MAX)
            # 如果 min_remaining < B_MIN，我们也必须跑，因为要把它清零
            batch_size = min(min_remaining, self.B_MAX)

            # d. 计算本轮时间参数 (对齐与同步)
            round_durations = [batch_size * tasks[idx]['duration'] for idx in round_tasks_indices]
            round_max_duration = max(round_durations)
            round_end_time = current_time + round_max_duration

            # e. 安排并更新
            for idx, my_duration in zip(round_tasks_indices, round_durations):
                t = tasks[idx]

                actual_start = round_end_time - my_duration
                actual_end = round_end_time

                t['shots_schedule'].append({
                    'start': actual_start,
                    'end': actual_end,
                    'num_shots': batch_size,
                    'batch_id': len(t['shots_schedule'])
                })

                t['remaining_shots'] -= batch_size
                t['age'] = 0
                pbar.update(batch_size)

            # f. 更新未被选中任务的年龄
            selected_set = set(round_tasks_indices)
            for idx in active_indices:
                if idx not in selected_set:
                    tasks[idx]['age'] += 1

            # g. 推进时间
            current_time = round_end_time

        pbar.close()

        return {
            "makespan": current_time,
            "tasks": tasks
        }
