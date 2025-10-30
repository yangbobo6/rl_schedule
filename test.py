# 文件: test.py

import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import matplotlib
import time

# 设置matplotlib后端为'Agg'，以避免在无GUI的环境中出错
matplotlib.use('Agg')

# --- 导入您项目中的核心模块 ---
# 假设您的代码结构使得这些导入能够正常工作
from environment import QuantumSchedulingEnv
from models.actor_critic import TransformerActorCritic
from models.gnn_encoder import GNNEncoder
from run import Hyperparameters  # 从run.py导入超参数类
from visualizer import plot_schedule

def test(run_dir: str, num_test_episodes: int = 100):
    """
    加载一个训练好的模型，在测试集 (test_pool) 上进行评估，并报告性能。
    
    Args:
        run_dir (str): 包含模型检查点 ('checkpoints/best_model.pth') 的训练结果目录。
        num_test_episodes (int): 用于评估的独立 aepisode 数量。
    """
    print(f"--- Starting Final Evaluation for Run: {run_dir} ---")
    
    # --- 1. 加载配置和设置环境 ---
    args = Hyperparameters()
    device = torch.device("cpu") # 评估通常在CPU上进行即可，速度足够快
    print(f"Using device: {device}")
    
    model_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Best model not found at '{model_path}'")
        return

    # 为本次测试创建一个专门的输出目录
    test_output_dir = os.path.join(run_dir, f"test_results_{int(time.time())}")
    gantt_charts_dir = os.path.join(test_output_dir, "gantt_charts")
    os.makedirs(gantt_charts_dir, exist_ok=True)
    print(f"Test outputs will be saved in: {test_output_dir}")


    # --- 2. 初始化所有必要的组件 (与run.py的初始化逻辑保持一致) ---
    gnn_model = GNNEncoder(
        node_feature_dim=args.GNN_NODE_DIM,
        hidden_dim=args.GNN_HIDDEN_DIM,
        output_dim=args.GNN_OUTPUT_DIM
    ).to(device)
    # GNN模型在预计算时使用，不需要训练
    gnn_model.eval() 

    gate_times = {
        'u3': 50, 'u': 50, 'p': 30, 'cx': 350, 'id': 30,
        'measure': 1000, 'rz': 30, 'sx': 40, 'x': 40
    }

    # 初始化环境
    env = QuantumSchedulingEnv(
        chip_size=args.CHIP_SIZE,
        num_tasks=args.NUM_TASKS,
        max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
        gnn_model=gnn_model,
        device=device,
        gate_times=gate_times,
        reward_weights=args.REWARD_WEIGHTS
    )

    # 初始化模型架构
    model = TransformerActorCritic(
        obs_space=env.observation_space,
        action_space_dim=env.num_qubits,
        d_model=args.D_MODEL,
        nhead=args.N_HEAD,
        num_encoder_layers=args.NUM_ENCODER_LAYERS
    ).to(device)

    # 加载训练好的最佳权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval() # 必须设置为评估模式！
    print("Model and environment initialized successfully.")

    # --- 3. 在测试集上运行评估循环 ---
    test_results = {
        "makespans": [], "fidelities": [], "swaps": [], "crosstalks": []
    }

    for episode_idx in tqdm(range(num_test_episodes), desc="Running Test Episodes"):
        # 告诉环境从'test'池中采样
        obs_dict, _ = env.reset(options={'pool': 'test'})
        
        # 在评估时，我们不计算梯度
        with torch.no_grad():
            for task_count in range(env.num_tasks):
                current_task = env.task_pool[task_count]
                num_logical_qubits = current_task.num_qubits
                placement_in_progress = {}

                for i in range(num_logical_qubits):
                    # 获取观察
                    if task_count == 0 and i == 0:
                        obs_dict_step = obs_dict
                    else:
                        obs_dict_step = env._get_obs(task_count, placement_in_progress)
                    
                    obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in obs_dict_step.items()}
                    
                    # 决策 (使用贪心策略，即选择概率最高的动作)
                    logits, _ = model(obs_tensor)

                    # --- 硬约束掩码逻辑 ---
                    placement_mask = torch.tensor(obs_dict_step["placement_mask"], dtype=torch.bool).to(device)
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
                    
                    action = torch.argmax(logits, dim=1) # 贪心选择
                    
                    # 更新放置
                    physical_qubit_id = env.idx_to_qubit_id[action.item()]
                    placement_in_progress[i] = physical_qubit_id
                
                # 提交计划
                final_mapping = placement_in_progress
                start_time = max(env.chip_model[pid].get_next_available_time() for pid in final_mapping.values()) if final_mapping else 0
                full_action = {"task_id": task_count, "mapping": final_mapping, "start_time": start_time}
                
                # 我们只关心env.step()返回的info字典
                _, _, terminated, _, info = env.step(full_action)

                # 累加物理指标
                test_results["fidelities"].append(info.get("final_fidelity", 0))
                test_results["swaps"].append(info.get("num_swaps", 0))
                test_results["crosstalks"].append(info.get("crosstalk_score", 0))
                
                if terminated:
                    break

        # 记录该episode的最终makespan
        if env.schedule_plan:
            makespan = max(t['end_time'] for t in env.schedule_plan)
            test_results["makespans"].append(makespan)

            # 为每个测试episode绘制并保存Gantt图
            baseline_makespan = sum(t.estimated_duration for t in env.task_pool.values()) if env.task_pool else 0
            plot_save_path = os.path.join(gantt_charts_dir, f"test_schedule_episode_{episode_idx}.png")
            plot_schedule(env.schedule_plan, args.CHIP_SIZE, baseline_makespan, save_path=plot_save_path)

    # --- 4. 报告最终结果 ---
    avg_makespan = np.mean(test_results["makespans"]) if test_results["makespans"] else 0
    avg_fidelity = np.mean(test_results["fidelities"]) if test_results["fidelities"] else 0
    avg_swaps = np.mean(test_results["swaps"]) if test_results["swaps"] else 0
    avg_crosstalk = np.mean(test_results["crosstalks"]) if test_results["crosstalks"] else 0
    
    results_str = (
        "\n" + "="*30 + " Final Test Results " + "="*30 + "\n"
        f"Evaluated on {num_test_episodes} episodes from the 'test' pool.\n\n"
        f"  - Average Makespan:         {avg_makespan:.2f} us\n"
        f"  - Average Task Fidelity:    {avg_fidelity:.4f}\n"
        f"  - Average SWAPs per Task:   {avg_swaps:.2f}\n"
        f"  - Average Crosstalk Score:  {avg_crosstalk:.4f}\n"
        + "="*82 + "\n"
    )
    print(results_str)

    # 将结果保存到文件中
    results_file_path = os.path.join(run_dir, "test_results.json")
    with open(results_file_path, 'w') as f:
        json.dump({
            "avg_makespan": avg_makespan,
            "avg_task_fidelity": avg_fidelity,
            "avg_swaps_per_task": avg_swaps,
            "avg_crosstalk_score": avg_crosstalk
        }, f, indent=4)
    print(f"Test results saved to {results_file_path}")

if __name__ == '__main__':
    # 使用 argparse 来接收命令行参数，这是一种很好的实践
    parser = argparse.ArgumentParser(description="Evaluate a trained quantum scheduling agent.")
    parser.add_argument(
        "--run_dir", 
        type=str, 
        required=True, 
        help="Path to the result directory of a training run (e.g., 'results/20250923-120000')."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of test episodes to run for evaluation."
    )
    
    script_args = parser.parse_args()
    
    test(script_args.run_dir, num_test_episodes=script_args.episodes)
