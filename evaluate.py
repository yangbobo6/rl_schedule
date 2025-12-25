# 文件: evaluate.py
import argparse
import time
from typing import Dict

from task_generate import TaskGenerator
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import json
import os
from torch.distributions import Categorical
from torch_geometric.data import Batch

# --- 导入所有自定义模块 ---
from environment import QuantumSchedulingEnv
from models.gnn_encoder import GNNEncoder, networkx_to_pyg_data
from models.actor_critic import TransformerActorCritic
from models.selector import SiameseGNNSelector
from run import Hyperparameters  # 从run.py导入超参数类
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, Layout
from visualizer import plot_all_schedules, plot_chip_snapshot
from models.hsamesq_scheduler import HSAMeSQScheduler


# ==============================================================================
# SECTION 1: 调度器算法的实现
# ==============================================================================

# --- 调度器基类 (可选，但推荐，用于规范接口) ---
class BaseScheduler:
    def __init__(self, name: str):
        self.name = name

    def schedule(self, env: QuantumSchedulingEnv) -> list:
        """
        接收一个配置好的环境，返回一个完整的调度计划。
        """
        raise NotImplementedError


def map_to_partition_sarbe(task, partition, env):
        """
        使用 Qiskit 的 SABRE 算法在指定分区内进行映射和路由。
        完全模拟 QuMC 的 mapping transition 过程。
        """
        # 1. 重建 Qiskit 线路 (从交互图恢复)
        # 我们需要一个真实的线路对象来运行 SABRE
        qc = QuantumCircuit(task.num_qubits)
        # 添加任务中的所有 CNOT 门 (逻辑连接)
        for u, v in task.interaction_graph.edges():
            qc.cx(u, v)

        # 2. 构建分区的局部拓扑 (Coupling Map)
        # 我们需要告诉 Qiskit 这个分区内部的连接情况
        # 将物理ID列表映射为局部索引: 0, 1, ..., k-1
        # partition 是 [pid1, pid2, ...]
        local_idx_to_physical_id = {i: pid for i, pid in enumerate(partition)}

        # 构建局部连接列表
        partition_coupling_list = []
        for i in range(len(partition)):
            for j in range(len(partition)):
                if i == j: continue
                u_pid = local_idx_to_physical_id[i]
                v_pid = local_idx_to_physical_id[j]

                # 如果这两个物理比特在真实芯片上相连
                if v_pid in env.chip_model[u_pid].connectivity:
                    partition_coupling_list.append([i, j])

        # 创建 Qiskit 的 CouplingMap 对象
        if not partition_coupling_list and task.num_qubits > 1:
            # 极端情况：如果分区不连通（理论上 GSP/QuMC 分区算法应避免这种情况）
            # 回退到之前的贪心逻辑或返回一个大惩罚
            return {}, 1000

        pm_coupling = CouplingMap(partition_coupling_list)

        # 3. 运行 Qiskit Transpile (SABRE 算法)
        # layout_method='sabre': 寻找最佳初始映射
        # routing_method='sabre': 插入最少的 SWAP
        # optimization_level=3: 开启最高级别的优化
        try:
            transpiled_qc = transpile(
                qc,
                coupling_map=pm_coupling,
                layout_method='sabre',
                routing_method='sabre',
                optimization_level=3
            )
        except Exception as e:
            print(f"SABRE mapping failed: {e}. Using fallback.")
            return {}, 1000

        # 4. 提取结果
        # a. 统计 SWAP 数量
        # count_ops() 返回字典，例如 {'cx': 10, 'swap': 2, ...}
        ops = transpiled_qc.count_ops()
        num_swaps = ops.get('swap', 0)

        # b. 提取映射关系
        mapping = {}

        if transpiled_qc.layout and transpiled_qc.layout.initial_layout:
            virtual_bits = transpiled_qc.layout.initial_layout.get_virtual_bits()
            for logical_qubit, local_physical_index in virtual_bits.items():
                # 使用 qc.qubits.index 获取索引
                try:
                    logic_id = qc.qubits.index(logical_qubit)
                except ValueError:
                    continue  # 这是一个不在原线路中的 ancilla 比特，忽略

                if logic_id < task.num_qubits:
                    true_physical_id = local_idx_to_physical_id[local_physical_index]
                    mapping[logic_id] = true_physical_id
        else:
            # Fallback
            print("Warning: No layout found, using trivial mapping.")
            for i in range(min(len(partition), task.num_qubits)):
                mapping[i] = partition[i]

        return mapping, num_swaps




class QuMCScheduler(BaseScheduler):
    def __init__(self, epsilon=100):
        super().__init__("QuMC-style Heuristic")
        self.epsilon = epsilon  # 保真度差异阈值

    def _qubit_fidelity_degree(self, qubit_idx, env, weight_lambda=1.0):
        """计算单个物理比特的 Fidelity Degree (公式2)"""
        # F_Degree = Sum( lambda * (1-E_cnot) + (1-E_readout) ) for neighbors
        degree_score = 0
        qubit = env.chip_model[env.idx_to_qubit_id[qubit_idx]]

        for neighbor_id in qubit.connectivity:
            # CNOT error = 1 - fidelity
            cnot_error = 1.0 - qubit.connectivity[neighbor_id]['fidelity_2q']
            # Readout error = 1 - fidelity
            readout_error = 1.0 - qubit.fidelity_readout

            # 简化版公式，直接用保真度计算
            # term = lambda * (1 - cnot_error) + (1 - readout_error)
            # term = lambda * fidelity_2q + fidelity_readout
            term = weight_lambda * qubit.connectivity[neighbor_id]['fidelity_2q'] + qubit.fidelity_readout
            degree_score += term

        return degree_score

    def _find_best_partition_qhsp(self, task, env, used_qubits_ids):
        """
        QHSP算法：基于 Fidelity Degree 的启发式分区。
        """
        available_indices = [i for i, qid in env.idx_to_qubit_id.items() if qid not in used_qubits_ids]
        if len(available_indices) < task.num_qubits:
            return None

        # 1. 筛选起始点 (Starting Points)
        # 按照物理度数排序，这里简化为按照 Fidelity Degree 排序
        sorted_start_points = sorted(
            available_indices,
            key=lambda i: self._qubit_fidelity_degree(i, env),
            reverse=True
        )
        # 只取前几个作为起点，模拟 heuristic
        starting_points = sorted_start_points[:len(available_indices) // 2 + 1]

        best_partition = None
        min_score = float('inf')

        for start_node_idx in starting_points:
            # 贪心增长分区
            partition = [start_node_idx]

            while len(partition) < task.num_qubits:
                # 在当前分区的邻居中，找 Fidelity Degree 最高的
                best_candidate = None
                best_candidate_score = -1

                # 遍历当前分区内的所有点，寻找它们的邻居
                # QHSP 策略：优先从当前分区中 Fidelity Degree 最高的点开始找邻居
                # 这里为了简化，遍历所有边界邻居

                potential_neighbors = set()
                for pid_idx in partition:
                    pid = env.idx_to_qubit_id[pid_idx]
                    for neighbor_id in env.chip_model[pid].connectivity:
                        n_idx = env.qubit_id_to_idx[neighbor_id]
                        if n_idx in available_indices and n_idx not in partition:
                            potential_neighbors.add(n_idx)

                if not potential_neighbors: break

                # 选择最好的邻居
                for n_idx in potential_neighbors:
                    score = self._qubit_fidelity_degree(n_idx, env)
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = n_idx

                if best_candidate is not None:
                    partition.append(best_candidate)
                else:
                    break

            # 如果成功构建了分区，计算其 Score
            if len(partition) == task.num_qubits:
                # 这里调用 QuMC 的评分函数
                score = self._calculate_partition_score_qumc(partition, task, env, used_qubits_ids)
                if score < min_score:
                    min_score = score
                    best_partition = [env.idx_to_qubit_id[i] for i in partition]

        return best_partition


    def _calculate_partition_score_qumc(self, partition_indices, task, env, used_qubits_ids_for_crosstalk):
        """
        计算分区的 Score (错误率)，包含串扰。
        模拟 partition_connectivity_error_rate_heuristic
        """
        partition_ids = [env.idx_to_qubit_id[i] for i in partition_indices]

        # 1. 计算 Avg_CNOT (含串扰)
        cnot_error_sum = 0
        num_links = 0

        # 模拟串扰：检查分区内的边是否受到外部已用比特的干扰
        # (这里我们简化处理，直接用我们现有的串扰模型)
        # 假设 crosstalk_within_partition = 0 (因为只算静态分数)
        # 但我们要加上与已存在任务的串扰

        crosstalk_penalty = 0.0
        # 遍历分区内所有物理连接
        for u_id in partition_ids:
            for v_id in partition_ids:
                if v_id in env.chip_model[u_id].connectivity:
                    # 基础错误
                    cnot_error_sum += (1.0 - env.chip_model[u_id].connectivity[v_id]['fidelity_2q'])

                    # 串扰错误：检查是否受到 used_qubits_ids_for_crosstalk 的干扰
                    # ... (此处可复用 env._calculate_crosstalk 的逻辑) ...
                    # 简单起见，我们假设如果邻居被占用了，就加一个固定惩罚系数
                    for neighbor_id in env.chip_model[u_id].connectivity:
                        if neighbor_id in used_qubits_ids_for_crosstalk:
                            crosstalk_penalty += 0.01  # 假设值

                    num_links += 1

        # 双向遍历会算两遍，除以2或者保持一致即可
        total_cnot_error = cnot_error_sum + crosstalk_penalty
        avg_cnot_error = total_cnot_error / num_links if num_links > 0 else 0.0

        # 2. Sum(R_Qi)
        readout_error_sum = sum(1.0 - env.chip_model[pid].fidelity_readout for pid in partition_ids)

        # 3. Final Score = Avg_CNOT * #CNOTs + Sum(R_Qi)
        num_cnots = task.interaction_graph.number_of_edges()
        score = avg_cnot_error * num_cnots + readout_error_sum
        return score


    def _map_to_partition(self, task, partition, env):
        """
        使用 Qiskit 的 SABRE 算法在指定分区内进行映射和路由。
        完全模拟 QuMC 的 mapping transition 过程。
        """
        # 1. 重建 Qiskit 线路 (从交互图恢复)
        # 我们需要一个真实的线路对象来运行 SABRE
        qc = QuantumCircuit(task.num_qubits)
        # 添加任务中的所有 CNOT 门 (逻辑连接)
        for u, v in task.interaction_graph.edges():
            qc.cx(u, v)

        # 2. 构建分区的局部拓扑 (Coupling Map)
        # 我们需要告诉 Qiskit 这个分区内部的连接情况
        # 将物理ID列表映射为局部索引: 0, 1, ..., k-1
        # partition 是 [pid1, pid2, ...]
        local_idx_to_physical_id = {i: pid for i, pid in enumerate(partition)}

        # 构建局部连接列表
        partition_coupling_list = []
        for i in range(len(partition)):
            for j in range(len(partition)):
                if i == j: continue
                u_pid = local_idx_to_physical_id[i]
                v_pid = local_idx_to_physical_id[j]

                # 如果这两个物理比特在真实芯片上相连
                if v_pid in env.chip_model[u_pid].connectivity:
                    partition_coupling_list.append([i, j])

        # 创建 Qiskit 的 CouplingMap 对象
        if not partition_coupling_list and task.num_qubits > 1:
            # 极端情况：如果分区不连通（理论上 GSP/QuMC 分区算法应避免这种情况）
            # 回退到之前的贪心逻辑或返回一个大惩罚
            return {}, 1000

        pm_coupling = CouplingMap(partition_coupling_list)

        # 3. 运行 Qiskit Transpile (SABRE 算法)
        # layout_method='sabre': 寻找最佳初始映射
        # routing_method='sabre': 插入最少的 SWAP
        # optimization_level=3: 开启最高级别的优化
        try:
            transpiled_qc = transpile(
                qc,
                coupling_map=pm_coupling,
                layout_method='sabre',
                routing_method='sabre',
                optimization_level=3
            )
        except Exception as e:
            print(f"SABRE mapping failed: {e}. Using fallback.")
            return {}, 1000

        # 4. 提取结果
        # a. 统计 SWAP 数量
        # count_ops() 返回字典，例如 {'cx': 10, 'swap': 2, ...}
        ops = transpiled_qc.count_ops()
        num_swaps = ops.get('swap', 0)

        # b. 提取映射关系
        mapping = {}

        if transpiled_qc.layout and transpiled_qc.layout.initial_layout:
            virtual_bits = transpiled_qc.layout.initial_layout.get_virtual_bits()
            for logical_qubit, local_physical_index in virtual_bits.items():
                # 使用 qc.qubits.index 获取索引
                try:
                    logic_id = qc.qubits.index(logical_qubit)
                except ValueError:
                    continue  # 这是一个不在原线路中的 ancilla 比特，忽略

                if logic_id < task.num_qubits:
                    true_physical_id = local_idx_to_physical_id[local_physical_index]
                    mapping[logic_id] = true_physical_id
        else:
            # Fallback
            print("Warning: No layout found, using trivial mapping.")
            for i in range(min(len(partition), task.num_qubits)):
                mapping[i] = partition[i]

        return mapping, num_swaps

    def _calculate_qumc_fidelity_score(self, task, mapping, env, crosstalk_within_batch, num_swaps):
        """
        修正版保真度计算：
        1. 利用了交互图的边权重 (CNOT次数)。
        2. 修复了物理不连通时忽略CNOT错误的Bug。
        3. 串扰被建模为对平均物理错误率的修正。
        """

        # --- 0. 准备工作：计算分区的物理基准 ---
        # 我们先看看这块物理区域的“底子”如何
        partition_nodes = list(mapping.values())
        phys_link_errors = []

        # 遍历分区内所有可能的物理连接
        for p_u in partition_nodes:
            for p_v in partition_nodes:
                if p_v in env.chip_model[p_u].connectivity:
                    # 收集该区域所有物理连线的错误率
                    phys_link_errors.append(1.0 - env.chip_model[p_u].connectivity[p_v]['fidelity_2q'])

        # 计算分区的平均物理错误率
        if phys_link_errors:
            avg_phys_error = np.mean(phys_link_errors)
            num_phys_links = len(phys_link_errors)
        else:
            avg_phys_error = 0.05  # 默认回退值
            num_phys_links = 1

        # --- 1. 融入串扰修正 ---
        # 串扰会提高这块区域的平均错误率
        # 既然 crosstalk_within_batch 是系数之和，我们将其平均分摊到每条物理连线上
        # 乘以 0.1 是为了将“系数和”转换为“概率增量” (经验值)
        crosstalk_penalty_rate = (crosstalk_within_batch * 0.1) / num_phys_links

        # 修正后的有效平均错误率 (Effective Error Rate)
        # 这将用于计算 SWAP 和 那些没有直接物理连接的 CNOT
        effective_avg_error = avg_phys_error + crosstalk_penalty_rate

        # --- 2. 计算 1Q 门错误 ---
        error_1q_sum = 0.0
        for pid in mapping.values():
            # 如果任务对象里有统计好的 num_1q_gates，就用它，否则估算
            # 假设 1Q 门也是均匀分布的
            num_1q = getattr(task, 'num_1q_gates', 0)
            if num_1q == 0: num_1q = task.depth * task.num_qubits  # 回退估算

            avg_gates_per_qubit = num_1q / task.num_qubits
            error_1q_sum += avg_gates_per_qubit * (1.0 - env.chip_model[pid].fidelity_1q)

        # --- 3. 计算 2Q 门错误 (逻辑门) ---
        error_2q_sum = 0.0

        # 遍历逻辑交互图的边
        # data 中包含了我们之前存入的 'weight' (CNOT次数)
        for u, v, data in task.interaction_graph.edges(data=True):
            count = data.get('weight', 1)  # 获取该边的CNOT次数

            p_u, p_v = mapping[u], mapping[v]

            if p_v in env.chip_model[p_u].connectivity:
                # Case A: 物理上直接连通
                # 使用这条物理连线特定的错误率 + 串扰修正
                specific_link_error = (1.0 - env.chip_model[p_u].connectivity[p_v]['fidelity_2q'])
                # 我们假设串扰是均匀的，或者你可以只加在这一项上
                current_error = specific_link_error + crosstalk_penalty_rate
                error_2q_sum += count * current_error
            else:
                # Case B: 物理上不连通 (Bug修复点)
                # 这些门不会消失，它们会在路由后的某个地方执行
                # 最佳估算就是使用当前分区的“有效平均错误率”
                error_2q_sum += count * effective_avg_error

        # --- 4. 计算 SWAP 错误 ---
        # SWAP = 3 * CNOT
        # SWAP 操作是在这块物理区域内进行的，所以使用 effective_avg_error
        error_swap_sum = num_swaps * 3 * effective_avg_error

        # --- 5. 最终汇总 ---
        total_error = error_1q_sum + error_2q_sum + error_swap_sum

        # 转换为保真度
        final_fidelity = np.exp(-total_error)

        return final_fidelity

    def _map_to_partition_old(self, task, partition, env):
        mapping = {}
        num_swaps = 0
        sorted_logical = sorted(task.interaction_graph.degree, key=lambda x: x[1], reverse=True)
        available_physical = list(partition)

        for logical_id, _ in sorted_logical:
            best_physical_id, min_cost = None, float('inf')

            for physical_id in available_physical:
                cost = sum(1 for pl, pp in mapping.items() if
                           task.interaction_graph.has_edge(logical_id, pl) and physical_id not in env.chip_model[
                               pp].connectivity)
                if cost < min_cost:
                    min_cost, best_physical_id = cost, physical_id

            if best_physical_id:
                mapping[logical_id] = best_physical_id
                available_physical.remove(best_physical_id)
                num_swaps += min_cost
        return mapping, num_swaps

    def schedule(self, env: QuantumSchedulingEnv) -> list:
        print(f"\n--- Running {self.name} ---")

        all_tasks = list(env.task_pool.values())
        # 1. 任务排序 (QuMC: 按 CNOT 密度)
        sorted_tasks = sorted(all_tasks, key=get_cnot_density, reverse=True)

        schedule_plan = []
        remaining_tasks = list(sorted_tasks)
        pbar = tqdm(total=len(all_tasks), desc="QuMC Heuristic")
        batch_start_time = 0.0

        while remaining_tasks:
            # =========================================================
            # PART 1: 并行度管理器 (Parallelism Manager)
            # =========================================================

            # 1. 初始候选批次：按顺序取尽可能多的任务，直到填满芯片
            candidate_batch = []
            qubits_needed = 0
            for task in remaining_tasks:
                if qubits_needed + task.num_qubits <= env.num_qubits:
                    candidate_batch.append(task)
                    qubits_needed += task.num_qubits
                else:
                    break  # 芯片放不下了

            final_batch_placements = []  # 存储最终决定的 (task, partition, mapping, swaps)
            batch_to_schedule_info = []  # 为了兼容后面的代码结构，我们也存一份这种格式

            # 2. 递减 K (并行任务数)，寻找满足保真度要求的最大并行度
            # 只要 candidate_batch 不为空，就尝试
            while candidate_batch:
                k_tasks = candidate_batch

                # a. 计算独立执行时的总分 (Score_independent)
                score_independent = 0
                for task in k_tasks:
                    # 独立分区：假设芯片是空的 (used_ids=set())
                    best_part = self._find_best_partition_qhsp(task, env, set())
                    if best_part:
                        # 计算分数 (无干扰)
                        s = self._calculate_partition_score_qumc(
                            [env.qubit_id_to_idx[pid] for pid in best_part], task, env, set()
                        )
                        score_independent += s
                    else:
                        score_independent += 1000  # 惩罚：找不到分区

                # b. 计算并行执行时的总分 (Score_parallel)
                score_parallel = 0
                temp_parallel_placements = []  # 暂存本轮尝试的结果
                temp_used_ids = set()
                possible_parallel = True

                for task in k_tasks:
                    # 并行分区：考虑已占用的比特 (temp_used_ids)
                    best_part = self._find_best_partition_qhsp(task, env, temp_used_ids)
                    if best_part:
                        # 计算分数：考虑来自 temp_used_ids 的干扰
                        s = self._calculate_partition_score_qumc(
                            [env.qubit_id_to_idx[pid] for pid in best_part], task, env, temp_used_ids
                        )
                        score_parallel += s
                        temp_parallel_placements.append((task, best_part))
                        temp_used_ids.update(best_part)
                    else:
                        possible_parallel = False
                        break  # 此 K 组合无法在物理上实现 (找不到空位)

                # c. 保真度检查 (Fidelity Check)
                if possible_parallel:
                    # 注意：Score是错误率，越小越好。
                    # 如果 并行错误 - 独立错误 < epslon，说明并行带来的损失可接受
                    delta_s = abs(score_parallel - score_independent)

                    if delta_s < self.epsilon:
                        # --- 成功！接受这个 K ---
                        # 为这些分区生成映射，并准备进入下一阶段
                        for task, partition in temp_parallel_placements:
                            mapping, num_swaps = self._map_to_partition(task, partition, env)

                            # 构建 info 字典，为了对接后面的代码
                            info = {'task': task, 'mapping': mapping, 'swaps': num_swaps}
                            batch_to_schedule_info.append(info)

                        break  # 成功找到合适的 K，跳出递减循环

                # --- 失败：减少 K，重试 ---
                # 移除密度最低的任务 (列表的最后一个)
                removed_task = candidate_batch.pop()

                # =========================================================
            # PART 2: 异常处理与状态更新
            # =========================================================

            # 死锁保护：如果连 K=1 都失败了 (candidate_batch为空)
            if not batch_to_schedule_info:
                if remaining_tasks:
                    # 强制调度第一个任务，避免死循环
                    # (通常是因为该任务即使单独跑也找不到合适分区，或者是芯片太小)
                    task = remaining_tasks[0]
                    # 尝试找一个最差的分区或者报错
                    # 这里简单处理：假设它至少能跑
                    print(f"Warning: Force scheduling task {task.id} due to failure.")

                    # 尝试强制找一个分区 (甚至可以是随机的)
                    # ... (省略强制逻辑，假设至少能找到一个)
                    # 如果真的找不到，就只能跳过它
                    remaining_tasks.pop(0)
                    pbar.update(1)
                    continue
                else:
                    break

            # 更新 remaining_tasks (移除已经决定调度的任务)
            scheduled_ids = {item['task'].id for item in batch_to_schedule_info}
            remaining_tasks = [t for t in remaining_tasks if t.id not in scheduled_ids]

            # =========================================================
            # PART 3: 计算指标并更新计划 (您提供的代码段)
            # =========================================================

            # a. 计算本批次所有任务的最终时长和批次的最大时长
            max_batch_duration = 0
            for item in batch_to_schedule_info:
                duration = item['task'].estimated_duration + item['swaps'] * (env.swap_penalty["duration"] / 1e3)
                item['duration'] = duration
                max_batch_duration = max(max_batch_duration, duration)

            # b. 为批次中的每个任务计算指标并添加到总计划
            for i, item in enumerate(batch_to_schedule_info):
                task = item['task']
                mapping = item['mapping']
                num_swaps = item['swaps']
                duration = item['duration']

                # 计算批次内的串扰
                crosstalk_within_batch = 0
                for j, other_item in enumerate(batch_to_schedule_info):
                    if i == j: continue
                    other_mapping = other_item['mapping']
                    for p_id1 in mapping.values():
                        for p_id2 in other_mapping.values():
                            if p_id2 in env.chip_model[p_id1].connectivity:
                                crosstalk_within_batch += env.chip_model[p_id1].connectivity[p_id2]['crosstalk_coeff']

                # 计算保真度
                fidelity = self._calculate_qumc_fidelity_score(task, mapping, env, crosstalk_within_batch, num_swaps)

                # --- 计算细粒度保真度指标 ---
                avg_f1q_used = np.mean([env.chip_model[pid].fidelity_1q for pid in mapping.values()])

                used_links_f2q = []
                for u, v in task.interaction_graph.edges():
                    p_u, p_v = mapping[u], mapping[v]
                    if p_v in env.chip_model[p_u].connectivity:
                        used_links_f2q.append(env.chip_model[p_u].connectivity[p_v]['fidelity_2q'])
                avg_f2q_used = np.mean(used_links_f2q) if used_links_f2q else 1.0

                schedule_plan.append({
                    "task_id": task.id,
                    "mapping": mapping,
                    "start_time": batch_start_time,
                    "end_time": batch_start_time + duration,
                    "num_swaps": num_swaps,
                    "final_fidelity": fidelity,
                    "crosstalk_score": crosstalk_within_batch,
                    "avg_f1q_used": avg_f1q_used,
                    "avg_f2q_used": avg_f2q_used
                })
                pbar.update(1)

            batch_start_time += max_batch_duration

        pbar.close()
        return schedule_plan



def get_cnot_density(task):
    """计算任务的CNOT密度"""
    num_cnots = task.interaction_graph.number_of_edges()
    num_qubits = task.num_qubits
    return num_cnots / num_qubits if num_qubits > 0 else 0


def calculate_crosstalk_test(self, mapping: Dict, start_time: float, end_time: float, schedule_plan) -> float:
        # 占位符：计算与已调度任务的串扰
        score = 0.0
        for scheduled in schedule_plan:
            # 检查时间重叠
            if max(start_time, scheduled['start_time']) < min(end_time, scheduled['end_time']):
                # 检查物理邻接
                for p_id1 in mapping.values():
                    for p_id2 in scheduled['mapping'].values():
                        if p_id2 in self.chip_model[p_id1].connectivity:
                            score += self.chip_model[p_id1].connectivity[p_id2]['crosstalk_coeff']
        return score

# ==============================================================================
# SECTION 2: RL调度器评估
# ==============================================================================
class RLScheduler(BaseScheduler):
    def __init__(self, model_path: str, gnn_selector_path: str, args: Hyperparameters):
        super().__init__("Reinforcement Learning Scheduler")
        self.model_path = model_path
        self.gnn_selector_path = gnn_selector_path
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _refine_with_sabre(self, task, rl_mapping, env):
        """
        使用 SABRE 在 RL 选定的物理比特集合内重新计算映射。
        """
        # 1. 获取 RL 选定的物理比特列表 (这就是我们的“区域”)
        partition_nodes = list(rl_mapping.values())  # list of (r, c)

        # 2. 构建这个区域的局部拓扑 (CouplingMap)
        # 为了让 SABRE 工作，我们需要将 (r,c) 映射为局部索引 0, 1, ... k-1
        phys_id_to_local_idx = {pid: i for i, pid in enumerate(partition_nodes)}
        local_idx_to_phys_id = {i: pid for i, pid in enumerate(partition_nodes)}

        coupling_list = []
        for i, u_pid in enumerate(partition_nodes):
            for j, v_pid in enumerate(partition_nodes):
                if i == j: continue
                # 如果这两个比特在真实芯片上相连，则在局部拓扑中添加边
                if v_pid in env.chip_model[u_pid].connectivity:
                    coupling_list.append([i, j])

        # 如果选出的区域连通性太差导致没有边，回退到 RL 映射
        if not coupling_list and task.num_qubits > 1:
            return rl_mapping, 100  # 惩罚值

        # 3. 构建 Qiskit 线路
        qc = QuantumCircuit(task.num_qubits)
        for u, v in task.interaction_graph.edges():
            qc.cx(u, v)

        # 4. 运行 SABRE
        # layout_method='sabre': 让 SABRE 决定在这个子图上逻辑比特该怎么放
        # routing_method='sabre': 计算动态 SWAP
        try:
            transpiled_qc = transpile(
                qc,
                coupling_map=CouplingMap(coupling_list),
                layout_method='sabre',
                routing_method='sabre',
                optimization_level=3
            )
        except Exception as e:
            print(f"SABRE refinement failed: {e}. Using original RL mapping.")
            return rl_mapping, 100

        # 5. 提取 SABRE 优化后的新映射和 SWAP 数
        real_num_swaps = transpiled_qc.count_ops().get('swap', 0)

        new_mapping = {}
        # 解析 layout
        if transpiled_qc.layout and transpiled_qc.layout.initial_layout:
            virtual_bits = transpiled_qc.layout.initial_layout.get_virtual_bits()
            for logical_qubit, local_phys_idx in virtual_bits.items():
                # 获取逻辑ID
                try:
                    # 尝试标准方法
                    logic_id = qc.qubits.index(logical_qubit)
                except:
                    continue  # 忽略 ancilla

                if logic_id < task.num_qubits:
                    # 将 SABRE 选出的局部索引转回全局物理ID
                    true_physical_id = local_idx_to_phys_id[local_phys_idx]
                    new_mapping[logic_id] = true_physical_id
        else:
            # 如果 transpile 没返回 layout，回退
            return rl_mapping, real_num_swaps

        return new_mapping, real_num_swaps


    def schedule(self, env: QuantumSchedulingEnv) -> list:
        print(f"\n--- Running {self.name} ---")

        gnn_selector = SiameseGNNSelector(chip_node_dim=1, task_node_dim=1, hidden_dim=64, embed_dim=32).to(self.device)
        gnn_selector.load_state_dict(torch.load(self.gnn_selector_path, map_location=self.device))
        gnn_selector.eval()

        rl_placer = TransformerActorCritic(
            env.observation_space, env.num_qubits,
            d_model=self.args.D_MODEL, nhead=self.args.N_HEAD, num_encoder_layers=self.args.NUM_ENCODER_LAYERS
        ).to(self.device)
        rl_placer.load_state_dict(torch.load(self.model_path, map_location=self.device))
        rl_placer.eval()

        # env.reset()

        with torch.no_grad():
            for _ in tqdm(range(env.num_tasks), desc="RL Scheduler"):
                candidate_tasks = env.get_candidate_tasks()
                if not candidate_tasks: break

                chip_graph_data = env.get_chip_state_graph_data()
                task_graph_data_list = [networkx_to_pyg_data(t.interaction_graph) for t in candidate_tasks]

                chip_batch = Batch.from_data_list([chip_graph_data] * len(candidate_tasks)).to(self.device)
                task_batch = Batch.from_data_list(task_graph_data_list).to(self.device)
                scores = gnn_selector(chip_batch, task_batch).cpu().numpy().flatten()

                chosen_task_local_index = np.argmax(scores)
                chosen_task = candidate_tasks[chosen_task_local_index]
                task_id = chosen_task.id

                current_task = env.task_pool[task_id]
                placement_in_progress = {}
                for i in range(current_task.num_qubits):
                    obs_dict_step = env._get_obs(task_id, placement_in_progress)
                    obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) for k, v in
                                  obs_dict_step.items()}
                    logits, _ = rl_placer(obs_tensor)

                    placement_mask = torch.tensor(obs_dict_step["placement_mask"], dtype=torch.bool).to(self.device)
                    # For simplicity, we omit the complex connectivity mask in evaluation,
                    # as the model should have learned to prefer connected placements. 加上
                    connectivity_mask = torch.ones_like(placement_mask)
                    task_graph = current_task.interaction_graph

                    if i == 0:
                        connectivity_mask = placement_mask.clone()
                    else:
                        valid_neighbors = set()
                        needs_connection = any(task_graph.has_edge(i, j) for j in placement_in_progress.keys())
                        if not needs_connection:
                            connectivity_mask = placement_mask.clone()
                        else:
                            for placed_logical_idx, placed_physical_id in placement_in_progress.items():
                                if task_graph.has_edge(i, placed_logical_idx):
                                    for neighbor_id in env.chip_model[placed_physical_id].connectivity:
                                        neighbor_idx = env.qubit_id_to_idx[neighbor_id]
                                        if not placement_mask[neighbor_idx]:
                                            valid_neighbors.add(neighbor_idx)
                            if valid_neighbors:
                                indices = torch.tensor(list(valid_neighbors), dtype=torch.long, device=device)
                                connectivity_mask.scatter_(0, indices, 0)
                            else:
                                connectivity_mask = placement_mask.clone()

                    final_mask = placement_mask | connectivity_mask
                    logits[0, final_mask] = -1e8

                    # 使用带温度的采样，既保留了贪心倾向，又允许一定的并行探索
                    temperature = 0.1  # 温度越低越接近贪心，越高越随机。0.1是一个不错的起点。
                    probs = torch.softmax(logits / temperature, dim=-1)
                    action = Categorical(probs).sample()

                    physical_qubit_id = env.idx_to_qubit_id[action.item()]
                    placement_in_progress[i] = physical_qubit_id

                # === 核心修改点 ===
                # placement_in_progress 是 RL 决定的原始映射
                # 调用 SABRE 在 RL 选定的区域内进行再优化
                refined_mapping, real_num_swaps = self._refine_with_sabre(current_task, placement_in_progress, env)
                # 使用优化后的映射和真实的 SWAP 数量
                final_mapping = refined_mapping

                # --- 4. 提交计划 (传入真实 SWAP) ---
                start_time = max(env.chip_model[pid].get_next_available_time() for pid in
                                 final_mapping.values()) if final_mapping else 0.0

                full_action = {
                    "task_id": task_id,
                    "mapping": final_mapping,
                    "start_time": start_time,
                    "override_swaps": real_num_swaps  # <--- 将真实值传递给 env.step
                }

                _, _, terminated, _, _ = env.step(full_action)

                if terminated: break
            return env.schedule_plan


class SequentialScheduler(BaseScheduler):
    """
    一个更智能的串行调度器。
    它按CNOT密度对任务进行排序，并在每一步为当前任务寻找芯片上可用的最佳分区进行映射。
    所有任务严格按顺序执行，一个接一个。
    """

    def __init__(self):
        super().__init__("Intelligent Sequential")

    def schedule(self, env: QuantumSchedulingEnv) -> list:
        print(f"\n--- Running {self.name} Scheduler ---")
        # env.reset()

        all_tasks = list(env.task_pool.values())

        # 1. 任务排序：按CNOT密度从高到低排序
        sorted_tasks = sorted(all_tasks, key=get_cnot_density, reverse=True)

        schedule_plan = []
        current_time = 0.0

        pbar = tqdm(total=len(sorted_tasks), desc="Intelligent Sequential")
        for task in sorted_tasks:

            # 2. 寻找最佳分区
            # 因为是串行执行，所以在为当前任务寻找分区时，整个芯片都是“可用”的。
            # 我们不需要传递 used_qubits_ids，或者传递一个空集合。
            best_partition = self._find_best_partition(task, env, used_qubits_ids=set())

            if not best_partition:
                print(f"Warning (Sequential): Could not find a valid partition for task {task.id}. Skipping.")
                pbar.update(1)
                continue

            # 3. 在找到的最佳分区内进行映射
            mapping, num_swaps = map_to_partition_sarbe(task, best_partition, env)

            # 4. 计算时间并添加到调度计划
            # 核心逻辑：下一个任务的开始时间 = 上一个任务的结束时间
            start_time = current_time
            duration = task.estimated_duration + num_swaps * (env.swap_penalty["duration"] / 1e3)
            end_time = start_time + duration

            # 计算物理指标
            final_fidelity = env._estimate_swaps_and_fidelity(task, mapping,0)[1]
            # 串行执行没有并行串扰
            crosstalk_score = 0

            # --- 计算细粒度保真度 ---
            avg_f1q_used = np.mean([env.chip_model[pid].fidelity_1q for pid in mapping.values()])

            used_links_f2q = []
            for u, v in task.interaction_graph.edges():
                p_u, p_v = mapping[u], mapping[v]
                if p_v in env.chip_model[p_u].connectivity:
                    used_links_f2q.append(env.chip_model[p_u].connectivity[p_v]['fidelity_2q'])
            avg_f2q_used = np.mean(used_links_f2q) if used_links_f2q else 1.0

            schedule_plan.append({
                "task_id": task.id, "mapping": mapping,
                "start_time": start_time, "end_time": end_time,
                "num_swaps": num_swaps,
                "final_fidelity": final_fidelity,
                "crosstalk_score": crosstalk_score,
                "avg_f1q_used": avg_f1q_used,
                "avg_f2q_used": avg_f2q_used
            })

            # 更新时间，为下一个任务做准备
            current_time = end_time
            pbar.update(1)

        pbar.close()
        return schedule_plan

    # --- 复用 QuMC 的辅助方法 ---
    # 这两个方法可以作为SequentialScheduler的私有方法，或者定义在全局
    def _find_best_partition(self, task, env, used_qubits_ids):
        # (此处的代码与您之前实现的 QuMCScheduler._find_best_partition 完全相同)
        best_partition = None
        best_score = -float('inf')

        for start_node_id in env.chip_model.keys():
            if start_node_id in used_qubits_ids:
                continue

            partition_nodes = {start_node_id}

            while len(partition_nodes) < task.num_qubits:
                boundary_neighbors = set()
                for node_in_partition in partition_nodes:
                    for neighbor_id in env.chip_model[node_in_partition].connectivity:
                        if neighbor_id not in partition_nodes and neighbor_id not in used_qubits_ids:
                            boundary_neighbors.add(neighbor_id)

                if not boundary_neighbors: break

                best_neighbor = max(boundary_neighbors, key=lambda nid: env.chip_model[nid].fidelity_1q, default=None)

                if best_neighbor:
                    partition_nodes.add(best_neighbor)
                else:
                    break

            if len(partition_nodes) == task.num_qubits:
                partition_score = sum(env.chip_model[nid].fidelity_1q for nid in partition_nodes)
                if partition_score > best_score:
                    best_score = partition_score
                    best_partition = list(partition_nodes)
        return best_partition

    def _map_to_partition(self, task, partition, env):
        # (QuMCScheduler._map_to_partition 完全相同)
        mapping, num_swaps = {}, 0
        sorted_logical = sorted(task.interaction_graph.degree, key=lambda x: x[1], reverse=True)
        available_physical = list(partition)

        for logical_id, _ in sorted_logical:
            best_physical_id, min_cost = None, float('inf')

            for physical_id in available_physical:
                cost = sum(1 for pl, pp in mapping.items() if
                           task.interaction_graph.has_edge(logical_id, pl) and physical_id not in env.chip_model[
                               pp].connectivity)
                if cost < min_cost:
                    min_cost, best_physical_id = cost, physical_id

            if best_physical_id:
                mapping[logical_id] = best_physical_id
                available_physical.remove(best_physical_id)
                num_swaps += min_cost
        return mapping, num_swaps


import itertools  # 需要导入itertools

class GreedyScheduler(BaseScheduler):
    """
    一个基于GSP (Greedy Sub-graph Partition) 思想的并行调度器。
    1. 按CNOT密度对任务排序。
    2. 分批次调度，尝试在每个批次中放入尽可能多的任务。
    3. 在为任务分配资源时，遍历所有可能的连通子图（分区），
       并选择一个成本分数最低的分区。
    """

    def __init__(self):
        super().__init__("GSP-based Greedy")

    def _calculate_partition_score(self, partition_nodes, task, env):
        """
        计算一个给定分区的成本分数，模拟论文中的Score_g。
        Score = L + Avg_CNOT * #CNOTs + Sum(R_Qi)
        """
        partition_graph = env.hardware_graph.subgraph(partition_nodes)

        # 1. 计算直径 (L)
        try:
            diameter = nx.diameter(partition_graph.to_undirected())
        except nx.NetworkXError:  # 如果图不连通
            return float('inf')

        # 2. 计算分区内的平均CNOT错误率 (Avg_CNOT)
        cnot_error_sum = 0
        num_links = 0
        for u, v in partition_graph.edges():
            link_fid = env.chip_model[u].connectivity.get(v, {}).get('fidelity_2q', 0.95)
            cnot_error_sum += (1.0 - link_fid)
            num_links += 1
        avg_cnot_error = cnot_error_sum / num_links if num_links > 0 else 1.0

        # 3. 计算分区内的总读出错误率 (Sum(R_Qi))
        sum_readout_error = sum(1.0 - env.chip_model[nid].fidelity_readout for nid in partition_nodes)

        # 4. 获取任务的CNOT数量
        num_task_cnots = task.interaction_graph.number_of_edges()

        # 综合分数 (越小越好)
        score = diameter + avg_cnot_error * num_task_cnots + sum_readout_error
        return score

    def _find_best_partition_gsp(self, task, env, used_qubits_ids):
        """
        遍历所有可能的连通子图，找到分数最低的最佳分区。
        """
        best_partition = None
        min_score = float('inf')

        available_nodes = [nid for nid in env.chip_model.keys() if nid not in used_qubits_ids]
        if len(available_nodes) < task.num_qubits:
            return None

        # 遍历所有可能的、与任务大小相同的物理比特组合
        for partition_candidate_nodes in itertools.combinations(available_nodes, task.num_qubits):
            # 检查这个组合是否在物理上是连通的
            subgraph = env.hardware_graph.subgraph(partition_candidate_nodes)
            if not nx.is_connected(subgraph.to_undirected()):
                continue

            # 计算这个候选分区的分数
            score = self._calculate_partition_score(partition_candidate_nodes, task, env)

            if score < min_score:
                min_score = score
                best_partition = list(partition_candidate_nodes)

        return best_partition

    def schedule(self, env: QuantumSchedulingEnv) -> list:
        print(f"\n--- Running {self.name} ---")
        # env.reset()

        all_tasks = list(env.task_pool.values())
        sorted_tasks = sorted(all_tasks, key=get_cnot_density, reverse=True)

        schedule_plan = []
        remaining_tasks = list(sorted_tasks)
        pbar = tqdm(total=len(all_tasks), desc=self.name)
        batch_start_time = 0.0

        while remaining_tasks:
            batch_to_schedule, temp_used_qubits, tasks_for_next_round = [], set(), []

            # 尝试将剩余任务装入当前并行批次
            for task in remaining_tasks:
                # 使用GSP算法在剩余的比特上寻找最佳分区
                best_partition = self._find_best_partition_gsp(task, env, temp_used_qubits)

                if best_partition:
                    # 在分区内进行映射 (可以复用之前的逻辑)
                    mapping, num_swaps = map_to_partition_sarbe(task, best_partition, env)
                    batch_to_schedule.append({'task': task, 'mapping': mapping, 'swaps': num_swaps})
                    temp_used_qubits.update(best_partition)
                else:
                    # 如果找不到分区，留到下一批次
                    tasks_for_next_round.append(task)

            if not batch_to_schedule:
                if remaining_tasks:
                    print(f"Warning: GSP Greedy couldn't place {len(remaining_tasks)} tasks.")
                break

            # 调度这个批次的所有任务
            max_batch_duration = 0
            batch_plan_items = []
            for item in batch_to_schedule:
                task, mapping, num_swaps = item['task'], item['mapping'], item['swaps']
                duration = task.estimated_duration + num_swaps * (env.swap_penalty["duration"] / 1e3)
                max_batch_duration = max(max_batch_duration, duration)
                batch_plan_items.append({
                    'task': task, 'mapping': mapping, 'num_swaps': num_swaps, 'duration': duration
                })

                # 现在，为批次中的每个任务计算串扰
                for i, item in enumerate(batch_plan_items):
                    task = item['task']
                    mapping = item['mapping']
                    duration = item['duration']

                    # 计算与“本批次内其他任务”的串扰
                    crosstalk_within_batch = 0
                    for j, other_item in enumerate(batch_plan_items):
                        if i == j: continue
                        other_mapping = other_item['mapping']
                        # 检查时间重叠 (在本批次内，时间总是重叠的)
                        # 检查物理邻接
                        for p_id1 in mapping.values():
                            for p_id2 in other_mapping.values():
                                if p_id2 in env.chip_model[p_id1].connectivity:
                                    crosstalk_within_batch += env.chip_model[p_id1].connectivity[p_id2]['crosstalk_coeff']

                    avg_f1q_used = np.mean([env.chip_model[pid].fidelity_1q for pid in mapping.values()])

                    used_links_f2q = []
                    for u, v in task.interaction_graph.edges():
                        p_u, p_v = mapping[u], mapping[v]
                        if p_v in env.chip_model[p_u].connectivity:
                            used_links_f2q.append(env.chip_model[p_u].connectivity[p_v]['fidelity_2q'])
                    avg_f2q_used = np.mean(used_links_f2q) if used_links_f2q else 1.0

                    schedule_plan.append({
                        "task_id": task.id,
                        "mapping": mapping,
                        "start_time": batch_start_time,
                        "end_time": batch_start_time + duration,
                        "num_swaps": item['num_swaps'],
                        "final_fidelity": env._estimate_swaps_and_fidelity(task, mapping,crosstalk_within_batch)[1],
                        "crosstalk_score": crosstalk_within_batch,
                        "avg_f1q_used": avg_f1q_used,
                        "avg_f2q_used": avg_f2q_used
                    })
                pbar.update(1)

            batch_start_time += max_batch_duration
            remaining_tasks = tasks_for_next_round

        pbar.close()
        return schedule_plan

    def _map_to_partition(self, task, partition, env):
        # (这个方法的代码与 SequentialScheduler 中的完全相同)
        mapping, num_swaps = {}, 0
        sorted_logical = sorted(task.interaction_graph.degree, key=lambda x: x[1], reverse=True)
        available_physical = list(partition)

        for logical_id, _ in sorted_logical:
            best_physical_id, min_cost = None, float('inf')

            for physical_id in available_physical:
                cost = sum(1 for pl, pp in mapping.items() if
                           task.interaction_graph.has_edge(logical_id, pl) and physical_id not in env.chip_model[
                               pp].connectivity)
                if cost < min_cost:
                    min_cost, best_physical_id = cost, physical_id

            if best_physical_id:
                mapping[logical_id] = best_physical_id
                available_physical.remove(best_physical_id)
                num_swaps += min_cost
        return mapping, num_swaps

# ==============================================================================
# SECTION 2: 主评估逻辑
# ==============================================================================

def calculate_metrics(schedule_plan: list) -> dict:
    if not schedule_plan:
        return {"makespan": float('inf'), "avg_swaps": float('inf'), "avg_fidelity": 0, "total_crosstalk": float('inf')}

    makespan = max(item['end_time'] for item in schedule_plan) if schedule_plan else 0
    avg_swaps = np.mean([item['num_swaps'] for item in schedule_plan])
    avg_fidelity = np.mean([item['final_fidelity'] for item in schedule_plan])
    total_crosstalk = sum(item['crosstalk_score'] for item in schedule_plan)
    avg_f1q = np.mean([item.get('avg_f1q_used', 0) for item in schedule_plan])
    avg_f2q = np.mean([item.get('avg_f2q_used', 0) for item in schedule_plan])
    return {"makespan": makespan, "avg_swaps": avg_swaps, "avg_fidelity": avg_fidelity,
            "total_crosstalk": total_crosstalk, "avg_f1q_used": avg_f1q, "avg_f2q_used": avg_f2q}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and compare quantum scheduling algorithms.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained RL placer model checkpoint (e.g., best_model.pth).")
    parser.add_argument("--gnn_selector_path", type=str, default="models/gnn_selector_v0.pth",
                        help="Path to the pre-trained GNN selector model.")
    script_args = parser.parse_args()
    args = Hyperparameters()

    print("Creating a fixed evaluation environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_for_task_gen = GNNEncoder(args.GNN_NODE_DIM, args.GNN_HIDDEN_DIM, args.GNN_OUTPUT_DIM).to(device)
    gate_times = {'u3': 50, 'u': 50, 'p': 30, 'cx': 350, 'id': 30, 'measure': 1000, 'rz': 30, 'sx': 40, 'x': 40}

    temp_env = QuantumSchedulingEnv(
        chip_size=args.CHIP_SIZE, num_tasks=args.NUM_TASKS, max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
        gnn_model=gnn_for_task_gen, device=device, gate_times=gate_times, reward_weights=args.REWARD_WEIGHTS
    )
    temp_task_generator = TaskGenerator(gate_times, gnn_for_task_gen, device)
    fixed_task_pool = temp_task_generator.get_episode_tasks(args.NUM_TASKS, pool='test')

    print("\n" + "=" * 20 + " EVALUATION TASK SET " + "=" * 20)
    task_ids = sorted(list(fixed_task_pool.keys()))
    # ... (打印任务信息) ...
    for tid in task_ids:
        task = fixed_task_pool[tid]
        print(f"  - Task {tid}: {task.num_qubits} qubits, {task.depth} depth, {task.interaction_graph.number_of_edges()} CNOTs")
    print("="*63)
    # --- 实例化所有要对比的调度器 ---
    schedulers_to_run = [
        # SequentialScheduler(),
        # GreedyScheduler(),
        QuMCScheduler(),
        RLScheduler(
            model_path=script_args.model_path,
            gnn_selector_path=script_args.gnn_selector_path,
            args=args
        )
    ]

    # 设定评估用的 Shots 数量
    SHOTS = 1000

    results = {}
    plans = {}

    # 临时变量，用于保存 RL 的环境实例，供 MAMS 使用
    rl_env_instance = None

    for scheduler in schedulers_to_run:
        env = QuantumSchedulingEnv(
            chip_size=args.CHIP_SIZE, num_tasks=args.NUM_TASKS, max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
            gnn_model=gnn_for_task_gen, device=device, gate_times=gate_times, reward_weights=args.REWARD_WEIGHTS
        )

        env.reset(fixed_task_pool=fixed_task_pool)
        plan = scheduler.schedule(env)
        results[scheduler.name] = calculate_metrics(plan)
        plans[scheduler.name] = plan

        # 保存 RL 的环境实例
        if scheduler.name == "Reinforcement Learning Scheduler":
            rl_env_instance = env

    rl_plan = plans["Reinforcement Learning Scheduler"]
    # 从 results 字典中获取 rl_results
    rl_results = results["Reinforcement Learning Scheduler"]

    # --- H-SAMeS-Q 后处理优化 ---
    if "Reinforcement Learning Scheduler" in plans and rl_env_instance:
        print("\n" + "=" * 20 + " H-SAMeS-Q OPTIMIZATION " + "=" * 20)

        # 实例化调度器
        # 可以通过 config 调整微批参数
        hsamesq = HSAMeSQScheduler(rl_env_instance, config={'B_MIN': 10, 'B_MAX': 50})

        # 运行调度
        SHOTS = 1000
        hsamesq_results = hsamesq.schedule(rl_plan, shots_per_task=SHOTS)

        print(f"\n--- H-SAMeS-Q Results ---")
        print(f"Original RL Makespan (1 shot): {rl_results['makespan']:.2f} us")
        print(f"Naive Linear Makespan ({SHOTS} shots): {rl_results['makespan'] * SHOTS:.2f} us")
        print(f"H-SAMeS-Q Optimized Makespan ({SHOTS} shots): {hsamesq_results['makespan']:.2f} us")

        improvement = (1 - hsamesq_results['makespan'] / (rl_results['makespan'] * SHOTS)) * 100
        print(f"Improvement over Linear: {improvement:.2f}%")

        # ... (保存结果逻辑) ...

        # 绘制 H-SAMeS-Q 的微观甘特图
        # 由于 shot 数太多，画出每一个 shot 不现实
        # 我们画出“微批 (Micro-batch)”级别的甘特图

        from visualizer import plot_micro_batch_schedule

        eval_dir = f"results/"

        plot_path = os.path.join(eval_dir, "hsamesq_schedule.png")
        plot_micro_batch_schedule(hsamesq_results['tasks'], args.CHIP_SIZE, save_path=plot_path)

    # --- 打印和保存对比结果 (这部分代码现在可以自动适应任意数量的调度器) ---
    print("\n\n" + "=" * 30 + " FINAL COMPARISON RESULTS " + "=" * 30)

    scheduler_names = list(results.keys())
    header = f"| {'Metric':<21} |" + "".join([f" {name:<25} |" for name in scheduler_names]) # 稍微调宽列宽
    print(header)
    print(f"|{'-' * 23}|" + "".join([f"{'-' * 27}|" for name in scheduler_names]))

    metrics_to_display = ["makespan", "avg_swaps", "avg_fidelity", "total_crosstalk"]
    for metric in metrics_to_display:
        row = f"| {metric:<21} |"
        for name in scheduler_names:
            value = results.get(name, {}).get(metric, 0)
            row += f" {value:>25.4f} |"
        print(row)
    print("=" * len(header))

    eval_timestamp = time.strftime("%Y%m%d-%H%M%S")
    eval_dir = f"results/evaluation_{eval_timestamp}"
    os.makedirs(eval_dir, exist_ok=True)

    # 保存JSON数据 (不变)
    with open(os.path.join(eval_dir, "comparison.json"), "w") as f:
        # 为了让JSON可读，我们需要处理mapping中的元组键
        # (这是一个好的实践)
        serializable_plans = {}
        for name, plan in plans.items():
            serializable_plans[name] = [
                {k: (str(v) if isinstance(v, dict) else v) for k, v in item.items()}
                for item in plan
            ]
    print(f"\nComparison data saved to {os.path.join(eval_dir, 'comparison.json')}")

    # 调用新的绘图函数
    plot_path = os.path.join(eval_dir, "schedule_comparison_all.png")
    plot_all_schedules(plans, args.CHIP_SIZE, save_path=plot_path)

    # --- 生成切面视图 ---
    # 选择一个有趣的时间点，比如 Makespan 的 1/4 或 1/3 处，那时通常并发度最高
    # 或者你可以手动指定，比如 5000
    if "Reinforcement Learning Scheduler" in plans:
        rl_plan = plans["Reinforcement Learning Scheduler"]
        total_makespan = results["Reinforcement Learning Scheduler"]["makespan"]
        snapshot_time = 5000  # 或者 total_makespan * 0.2

        snapshot_save_path = os.path.join(eval_dir, f"snapshot_{snapshot_time}us.png")
        plot_chip_snapshot(rl_plan, args.CHIP_SIZE, snapshot_time, save_path=snapshot_save_path)


def evaluate_batch(schedulers, base_env, num_episodes=50, chip_size=(6, 6), num_tasks=20):
    """
    运行多轮评估，收集统计数据。
    """
    print(f"\n=== Starting Batch Evaluation ({num_episodes} episodes) ===")

    # 存储所有结果: {scheduler_name: {metric: [val1, val2, ...]}}
    all_results = {name: {"makespan": [], "avg_swaps": [], "avg_fidelity": [], "total_crosstalk": []}
                   for name in [s.name for s in schedulers]}

    for episode_idx in tqdm(range(num_episodes), desc="Evaluating Episodes"):
        # 1. 为本轮生成一组固定的随机任务
        # 我们使用 base_env 的 task_generator 来生成，这会保证 ID 的唯一性
        # 注意：我们需要一种方法来强制所有环境在这一轮使用这一组任务
        # 最简单的方法是手动设置 seed，或者直接生成好任务字典传进去

        # 生成新的任务集
        episode_tasks = base_env.task_generator.get_episode_tasks(num_tasks, pool='test')

        for scheduler in schedulers:
            # 为每个调度器创建一个干净的环境
            # 注意：这里我们假设环境创建开销不大。如果很大，可以考虑 reset 重用
            env = QuantumSchedulingEnv(
                chip_size=chip_size, num_tasks=num_tasks, max_qubits_per_task=8,
                gnn_model=base_env.task_generator.gnn_model,  # 复用
                device=base_env.task_generator.device,  # 复用
                gate_times=base_env.gate_times,
                reward_weights={}
            )

            # 强制使用相同的任务集
            env.task_pool = episode_tasks

            # 运行调度
            try:
                plan = scheduler.schedule(env)
                metrics = calculate_metrics(plan)

                # 记录指标
                for k, v in metrics.items():
                    # 过滤掉无效值
                    if v != float('inf'):
                        all_results[scheduler.name][k].append(v)
            except Exception as e:
                print(f"Error in {scheduler.name} at episode {episode_idx}: {e}")

    return all_results


