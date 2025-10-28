# 文件: evaluate.py
import time

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
# SECTION 1: QuMC风格的启发式调度器
# ==============================================================================

def get_cnot_density(task):
    """计算任务的CNOT密度"""
    num_cnots = task.interaction_graph.number_of_edges()
    num_qubits = task.num_qubits
    return num_cnots / num_qubits if num_qubits > 0 else 0


def find_best_partition_heuristic(task, env, used_qubits_indices):
    """简化的启发式分区查找器"""
    best_partition = None
    best_score = -float('inf')

    for start_node_idx in range(env.num_qubits):
        if start_node_idx in used_qubits_indices:
            continue
        start_node_id = env.idx_to_qubit_id[start_node_idx]

        partition_nodes = {start_node_id}
        current_used_indices = {start_node_idx}

        while len(partition_nodes) < task.num_qubits:
            boundary_neighbors = set()
            for node_in_partition in partition_nodes:
                for neighbor_id in env.chip_model[node_in_partition].connectivity:
                    if neighbor_id not in partition_nodes and env.qubit_id_to_idx[
                        neighbor_id] not in used_qubits_indices:
                        boundary_neighbors.add(neighbor_id)

            if not boundary_neighbors: break

            best_neighbor = max(boundary_neighbors, key=lambda nid: env.chip_model[nid].fidelity_1q, default=None)

            if best_neighbor:
                partition_nodes.add(best_neighbor)
                current_used_indices.add(env.qubit_id_to_idx[best_neighbor])
            else:
                break

        if len(partition_nodes) == task.num_qubits:
            partition_score = sum(env.chip_model[nid].fidelity_1q for nid in partition_nodes)
            if partition_score > best_score:
                best_score = partition_score
                best_partition = list(partition_nodes)

    return best_partition


def map_task_to_partition(task, partition, env):
    """在给定分区内进行贪心映射"""
    mapping = {}
    num_swaps = 0

    sorted_logical = sorted(task.interaction_graph.degree, key=lambda x: x[1], reverse=True)
    available_physical = list(partition)

    for logical_id, _ in sorted_logical:
        best_physical_id = None
        min_cost = float('inf')

        for physical_id in available_physical:
            cost = 0
            for placed_logic, placed_phys in mapping.items():
                if task.interaction_graph.has_edge(logical_id, placed_logic):
                    if physical_id not in env.chip_model[placed_phys].connectivity:
                        cost += 1

            if cost < min_cost:
                min_cost = cost
                best_physical_id = physical_id

        if best_physical_id:
            mapping[logical_id] = best_physical_id
            available_physical.remove(best_physical_id)
            num_swaps += min_cost

    return mapping, num_swaps


def run_qumc_heuristic_scheduler(env: QuantumSchedulingEnv):
    """运行简化的QuMC启发式调度算法"""
    print("\n--- Running QuMC-style Heuristic Scheduler ---")

    tasks = list(env.task_pool.values())
    sorted_tasks = sorted(tasks, key=get_cnot_density, reverse=True)

    schedule_plan = []
    qubit_release_times = np.zeros(env.num_qubits)
    used_qubits_indices = set()

    for task in tqdm(sorted_tasks, desc="QuMC Heuristic"):
        best_partition = find_best_partition_heuristic(task, env, used_qubits_indices)
        if not best_partition:
            print(f"Warning: Could not find a valid partition for task {task.id}")
            continue

        mapping, num_swaps = map_task_to_partition(task, best_partition, env)

        partition_indices = [env.qubit_id_to_idx[pid] for pid in best_partition]
        start_time = np.max(qubit_release_times[partition_indices])
        duration = task.estimated_duration + num_swaps * (env.swap_penalty["duration"] / 1e3)
        end_time = start_time + duration

        for idx in partition_indices:
            qubit_release_times[idx] = end_time
        used_qubits_indices.update(partition_indices)
        crosstalk_score = env._calculate_crosstalk(mapping, start_time, end_time)
        # 然后再将新任务添加到 env.schedule_plan
        new_schedule_item = {
            "task_id": task.id, "mapping": mapping,
            "start_time": start_time, "end_time": end_time,
            "num_swaps": num_swaps,
            "final_fidelity": env._estimate_swaps_and_fidelity(task, mapping)[1],
            "crosstalk_score": crosstalk_score
        }
        schedule_plan.append(new_schedule_item)

        # 为了让下一次循环的_calculate_crosstalk能看到最新状态，我们需要更新env的状态
        env.schedule_plan = schedule_plan

    return schedule_plan


# ==============================================================================
# SECTION 2: RL调度器评估
# ==============================================================================

def run_rl_scheduler(env: QuantumSchedulingEnv, model_path: str, gnn_selector_path: str, args: Hyperparameters):
    """运行我们训练好的RL调度器 (GNN选择 + Transformer放置)"""
    print("\n--- Running RL Scheduler ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    gnn_selector = SiameseGNNSelector(
        chip_node_dim=1, task_node_dim=1,
        hidden_dim=64, embed_dim=32
    ).to(device)
    gnn_selector.load_state_dict(torch.load(gnn_selector_path, map_location=device))
    gnn_selector.eval()

    rl_placer = TransformerActorCritic(
        env.observation_space, env.num_qubits,
        d_model=args.D_MODEL, nhead=args.N_HEAD,
        num_encoder_layers=args.NUM_ENCODER_LAYERS
    ).to(device)
    rl_placer.load_state_dict(torch.load(model_path, map_location=device))
    rl_placer.eval()

    # 2. 运行一个完整的Episode进行评估
    # obs_dict, _ = env.reset()

    for _ in tqdm(range(env.num_tasks), desc="RL Scheduler"):
        # a. GNN选择任务
        candidate_tasks = env.get_candidate_tasks()
        if not candidate_tasks: break

        chip_graph_data = env.get_chip_state_graph_data()
        task_graph_data_list = [networkx_to_pyg_data(t.interaction_graph) for t in candidate_tasks]

        with torch.no_grad():
            chip_batch = Batch.from_data_list([chip_graph_data] * len(candidate_tasks)).to(device)
            task_batch = Batch.from_data_list(task_graph_data_list).to(device)
            scores = gnn_selector(chip_batch, task_batch).cpu().numpy().flatten()

        chosen_task_local_index = np.argmax(scores)
        chosen_task = candidate_tasks[chosen_task_local_index]
        task_id = chosen_task.id

        # b. RL放置任务
        current_task = env.task_pool[task_id]
        placement_in_progress = {}
        for i in range(current_task.num_qubits):
            obs_dict = env._get_obs(task_id, placement_in_progress)
            obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in obs_dict.items()}

            with torch.no_grad():
                logits, _ = rl_placer(obs_tensor)

            final_mask = torch.tensor(obs_dict["placement_mask"], dtype=torch.bool).to(device)
            logits[0, final_mask] = -1e8

            # 直接在 logits 上取最大值的索引
            action = logits.argmax(dim=-1)

            physical_qubit_id = env.idx_to_qubit_id[action.item()]
            placement_in_progress[i] = physical_qubit_id

        # c. 提交给环境
        final_mapping = placement_in_progress
        start_time = 0.0
        if final_mapping:
            # 这个计算是正确的，它依赖于 env.chip_model 的状态
            start_time = max(env.chip_model[pid].get_next_available_time() for pid in final_mapping.values())

        full_action = {"task_id": task_id, "mapping": final_mapping, "start_time": start_time}

        # 调试打印
        print(f"  - Placing Task {task_id}. Mapping: {final_mapping}")
        release_times = {pid: env.chip_model[pid].get_next_available_time() for pid in final_mapping.values()}
        print(f"  - Release times for mapped qubits: {release_times}")
        print(f"  - Calculated Start Time: {start_time}")

        _, _, terminated, _, _ = env.step(full_action)

        if terminated:
            break

    return env.schedule_plan


# ==============================================================================
# SECTION 3: 主评估流程
# ==============================================================================

def calculate_metrics(schedule_plan: list) -> dict:
    """从一个调度计划中计算所有关键指标"""
    if not schedule_plan:
        return {"makespan": float('inf'), "avg_swaps": float('inf'), "avg_fidelity": 0, "total_crosstalk": float('inf')}

    makespan = max(item['end_time'] for item in schedule_plan)
    avg_swaps = np.mean([item['num_swaps'] for item in schedule_plan])
    avg_fidelity = np.mean([item['final_fidelity'] for item in schedule_plan])
    total_crosstalk = sum(item['crosstalk_score'] for item in schedule_plan)
    return {"makespan": makespan, "avg_swaps": avg_swaps, "avg_fidelity": avg_fidelity,
            "total_crosstalk": total_crosstalk}


if __name__ == '__main__':
    args = Hyperparameters()

    # --- 1. 创建一个固定的评估环境 ---
    print("Creating a fixed evaluation environment...")

    # 初始化GNN和gate_times等共享资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_for_task_gen = GNNEncoder(
        node_feature_dim=args.GNN_NODE_DIM,
        hidden_dim=args.GNN_HIDDEN_DIM,
        output_dim=args.GNN_OUTPUT_DIM
    ).to(device)

    gate_times = {
        'u3': 50, 'u': 50, 'p': 30, 'cx': 350, 'id': 30,
        'measure': 1000, 'rz': 30, 'sx': 40, 'x': 40
    }

    # 创建一个临时环境来生成一个固定的任务池，以保证公平对比
    temp_env = QuantumSchedulingEnv(
        chip_size=args.CHIP_SIZE,
        num_tasks=args.NUM_TASKS,
        max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
        gnn_model=gnn_for_task_gen,
        device=device,
        gate_times=gate_times,
        reward_weights=args.REWARD_WEIGHTS
    )
    # 强制构建任务池
    temp_env.task_generator.build_large_task_pool(100)
    # 获取一个固定的任务集用于本次评估
    fixed_task_pool = temp_env.task_generator.get_episode_tasks(args.NUM_TASKS)
    print(f"Evaluation will be performed on a fixed set of {len(fixed_task_pool)} tasks.")

    # --- 2. 运行 QuMC 启发式算法 ---
    # 创建一个干净的环境实例
    env_qumc = QuantumSchedulingEnv(
        chip_size=args.CHIP_SIZE, num_tasks=args.NUM_TASKS, max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
        gnn_model=gnn_for_task_gen, device=device, gate_times=gate_times, reward_weights=args.REWARD_WEIGHTS
    )
    env_qumc.task_pool = fixed_task_pool
    env_qumc.reset()  # 确保内部状态正确

    qumc_plan = run_qumc_heuristic_scheduler(env_qumc)
    qumc_results = calculate_metrics(qumc_plan)

    # --- 3. 运行 RL 调度器 ---
    # 创建另一个干净的环境实例
    env_rl = QuantumSchedulingEnv(
        chip_size=args.CHIP_SIZE, num_tasks=args.NUM_TASKS, max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
        gnn_model=gnn_for_task_gen, device=device, gate_times=gate_times, reward_weights=args.REWARD_WEIGHTS
    )
    env_rl.task_pool = fixed_task_pool

    rl_plan = run_rl_scheduler(
        env_rl,
        model_path="models/model_episode_49800.pth",  # <-- 请务必替换为您最好的RL模型路径
        gnn_selector_path="models/gnn_selector_v0.pth",
        args=args
    )
    rl_results = calculate_metrics(rl_plan)

    # --- 4. 打印、保存和可视化对比结果 ---
    print("\n\n" + "=" * 25 + " COMPARISON RESULTS " + "=" * 25)
    print(f"| Metric                | QuMC Heuristic | RL Scheduler   | Improvement (%) |")
    print(f"|-----------------------|----------------|----------------|-----------------|")

    # Makespan
    imp_mk = (qumc_results['makespan'] - rl_results['makespan']) / qumc_results['makespan'] * 100 if qumc_results[
                                                                                                         'makespan'] > 0 else 0
    print(
        f"| Makespan (us)         | {qumc_results['makespan']:>14.2f} | {rl_results['makespan']:>14.2f} | {imp_mk:>14.2f}% |")

    # SWAPs
    imp_sw = (qumc_results['avg_swaps'] - rl_results['avg_swaps']) / qumc_results['avg_swaps'] * 100 if qumc_results[
                                                                                                            'avg_swaps'] > 0 else 0
    print(
        f"| Avg SWAPs per Task    | {qumc_results['avg_swaps']:>14.2f} | {rl_results['avg_swaps']:>14.2f} | {imp_sw:>14.2f}% |")

    # Fidelity
    imp_fid = (rl_results['avg_fidelity'] - qumc_results['avg_fidelity']) / qumc_results['avg_fidelity'] * 100 if \
    qumc_results['avg_fidelity'] > 0 else 0
    print(
        f"| Avg Task Fidelity     | {qumc_results['avg_fidelity']:>14.4f} | {rl_results['avg_fidelity']:>14.4f} | {imp_fid:>14.2f}% |")

    # Crosstalk
    imp_ct = (qumc_results['total_crosstalk'] - rl_results['total_crosstalk']) / qumc_results[
        'total_crosstalk'] * 100 if qumc_results['total_crosstalk'] > 0 else 0
    print(
        f"| Total Crosstalk Score | {qumc_results['total_crosstalk']:>14.2f} | {rl_results['total_crosstalk']:>14.2f} | {imp_ct:>14.2f}% |")

    print("=" * 71)

    # 创建一个唯一的评估结果目录
    eval_timestamp = time.strftime("%Y%m%d-%H%M%S")
    eval_dir = f"results/evaluation_{eval_timestamp}"
    os.makedirs(eval_dir, exist_ok=True)

    # 保存结果到JSON文件
    comparison_data = {"qumc_heuristic": qumc_results, "rl_scheduler": rl_results}
    with open(os.path.join(eval_dir, "comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=4)
    print(f"\nComparison data saved to {os.path.join(eval_dir, 'comparison.json')}")

    # 保存对比图
    plot_path = os.path.join(eval_dir, "schedule_comparison.png")
    plot_comparison(rl_plan, qumc_plan, args.CHIP_SIZE, save_path=plot_path)
