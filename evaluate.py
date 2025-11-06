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
from visualizer import plot_comparison

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


class QuMCScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("QuMC-style Heuristic")

    def _find_best_partition(self, task, env, used_qubits_ids):
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
        env.reset()

        all_tasks = list(env.task_pool.values())
        sorted_tasks = sorted(all_tasks, key=get_cnot_density, reverse=True)

        schedule_plan = []
        remaining_tasks = list(sorted_tasks)
        pbar = tqdm(total=len(all_tasks), desc="QuMC Heuristic")
        batch_start_time = 0.0

        while remaining_tasks:
            batch_to_schedule, temp_used_qubits, tasks_for_next_round = [], set(), []

            for task in remaining_tasks:
                available_qubits_count = env.num_qubits - len(temp_used_qubits)
                if available_qubits_count < task.num_qubits:
                    tasks_for_next_round.append(task)
                    continue

                best_partition = self._find_best_partition(task, env, temp_used_qubits)

                if best_partition:
                    mapping, num_swaps = self._map_to_partition(task, best_partition, env)
                    batch_to_schedule.append({'task': task, 'mapping': mapping, 'swaps': num_swaps})
                    temp_used_qubits.update(best_partition)
                else:
                    tasks_for_next_round.append(task)

            if not batch_to_schedule:
                if remaining_tasks: print(
                    f"Warning: Deadlock in QuMC, {len(remaining_tasks)} tasks cannot be scheduled.")
                break

            max_batch_duration = 0
            current_batch_plan = []
            for item in batch_to_schedule:
                task, mapping, num_swaps = item['task'], item['mapping'], item['swaps']
                duration = task.estimated_duration + num_swaps * (env.swap_penalty["duration"] / 1e3)
                max_batch_duration = max(max_batch_duration, duration)
                current_batch_plan.append((task, mapping, num_swaps, duration))

            for task, mapping, num_swaps, duration in current_batch_plan:
                schedule_plan.append({
                    "task_id": task.id, "mapping": mapping,
                    "start_time": batch_start_time, "end_time": batch_start_time + duration,
                    "num_swaps": num_swaps,
                    "final_fidelity": env._estimate_swaps_and_fidelity(task, mapping)[1],
                    "crosstalk_score": env._calculate_crosstalk(mapping, batch_start_time, batch_start_time + duration)
                })
                pbar.update(1)

            batch_start_time += max_batch_duration
            remaining_tasks = tasks_for_next_round

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

        env.reset()

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

                    action = torch.argmax(logits, dim=1)
                    physical_qubit_id = env.idx_to_qubit_id[action.item()]
                    placement_in_progress[i] = physical_qubit_id

                final_mapping = placement_in_progress
                start_time = max(env.chip_model[pid].get_next_available_time() for pid in
                                 final_mapping.values()) if final_mapping else 0.0
                full_action = {"task_id": task_id, "mapping": final_mapping, "start_time": start_time}
                _, _, terminated, _, _ = env.step(full_action)

                if terminated: break
        return env.schedule_plan



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
    return {"makespan": makespan, "avg_swaps": avg_swaps, "avg_fidelity": avg_fidelity,
            "total_crosstalk": total_crosstalk}


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
        QuMCScheduler(),
        RLScheduler(
            model_path=script_args.model_path,
            gnn_selector_path=script_args.gnn_selector_path,
            args=args
        )
    ]

    results = {}
    plans = {}

    for scheduler in schedulers_to_run:
        env = QuantumSchedulingEnv(
            chip_size=args.CHIP_SIZE, num_tasks=args.NUM_TASKS, max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
            gnn_model=gnn_for_task_gen, device=device, gate_times=gate_times, reward_weights=args.REWARD_WEIGHTS
        )
        env.task_pool = fixed_task_pool

        plan = scheduler.schedule(env)
        results[scheduler.name] = calculate_metrics(plan)
        plans[scheduler.name] = plan

    # --- 打印和保存对比结果 ---
    print("\n\n" + "=" * 25 + " COMPARISON RESULTS " + "=" * 25)
    header = f"| {'Metric':<21} |" + "".join([f" {name:<14} |" for name in results.keys()])
    print(header)
    print(f"|{'-' * 23}|" + "".join([f"{'-' * 16}|" for name in results.keys()]))

    for metric in results[list(results.keys())[0]]:
        row = f"| {metric:<21} |"
        for name in results.keys():
            value = results[name][metric]
            row += f" {value:>14.4f} |"
        print(row)
    print("=" * len(header))

    # --- 保存和可视化 ---
    eval_timestamp = time.strftime("%Y%m%d-%H%M%S")
    eval_dir = f"results/evaluation_{eval_timestamp}"
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, "comparison.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nComparison data saved to {os.path.join(eval_dir, 'comparison.json')}")

    plot_path = os.path.join(eval_dir, "schedule_comparison.png")
    plot_comparison(plans.get("Reinforcement Learning Scheduler", []), plans.get("QuMC-style Heuristic", []),
                    args.CHIP_SIZE, save_path=plot_path)