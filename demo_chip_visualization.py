"""
量子芯片可视化演示脚本
展示如何使用QuantumChipVisualizer来分析和可视化量子芯片
"""

import torch
import os
from environment import QuantumSchedulingEnv
from models.gnn_encoder import GNNEncoder


def main():
    """演示量子芯片可视化功能"""
    print("=== Quantum Chip Visualization Demo ===\n")
    
    # 设置参数
    chip_size = (4, 4)  # 4x4的量子芯片
    num_tasks = 10
    max_qubits_per_task = 6
    
    # 创建结果目录
    results_dir = "results/chip_visualization"
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化GNN模型
    gnn_model = GNNEncoder(
        node_feature_dim=1,
        hidden_dim=32,
        output_dim=16
    ).to(device)
    gnn_model.eval()
    
    # 门时间配置
    gate_times = {
        'u3': 50, 'u': 50, 'p': 30, 'cx': 350, 'id': 30,
        'measure': 1000, 'rz': 30, 'sx': 40, 'x': 40
    }
    
    # 创建量子调度环境
    print("Creating quantum scheduling environment...")
    env = QuantumSchedulingEnv(
        chip_size=chip_size,
        num_tasks=num_tasks,
        max_qubits_per_task=max_qubits_per_task,
        gnn_model=gnn_model,
        device=device,
        gate_times=gate_times
    )
    
    print(f"Created {chip_size[0]}×{chip_size[1]} quantum chip with {len(env.chip_model)} qubits\n")
    
    # 1. 可视化芯片总览
    print("1. Generating chip overview visualization...")
    chip_overview_path = os.path.join(results_dir, "chip_overview.png")
    env.visualize_chip(chip_overview_path, show_values=True)
    
    # 2. 可视化连接矩阵
    print("2. Generating connectivity matrix...")
    connectivity_path = os.path.join(results_dir, "connectivity_matrix.png")
    env.visualize_connectivity(connectivity_path)
    
    # 3. 导出芯片统计信息
    print("3. Exporting chip statistics...")
    stats_path = os.path.join(results_dir, "chip_stats.json")
    stats = env.export_chip_stats(stats_path)
    
    print("\n=== Chip Statistics Summary ===")
    print(f"Chip Size: {stats['chip_size']}")
    print(f"Total Qubits: {stats['total_qubits']}")
    print(f"Average T1: {stats['avg_t1']:.2f} μs")
    print(f"Average T2: {stats['avg_t2']:.2f} μs")
    print(f"Average Single-Qubit Fidelity: {stats['avg_fidelity_1q']:.4f}")
    print(f"Average Readout Fidelity: {stats['avg_fidelity_readout']:.4f}")
    if 'avg_fidelity_2q' in stats:
        print(f"Average Two-Qubit Fidelity: {stats['avg_fidelity_2q']:.4f}")
    print(f"Total Connections: {stats['total_connections']}")
    print(f"Average Connectivity: {stats['avg_connectivity']:.2f}")
    
    # 4. 模拟一些调度并可视化调度状态
    print("\n4. Simulating scheduling and visualizing states...")
    
    # 重置环境
    obs, _ = env.reset()
    
    # 模拟几个简单的调度决策
    for step in range(min(3, num_tasks)):
        # 获取当前任务
        current_task = env.task_pool[step]
        print(f"\nScheduling Task {step}: {current_task.num_qubits} qubits, "
              f"duration: {current_task.estimated_duration:.2f} μs")
        
        # 简单的贪心映射：选择质量最好且可用的量子比特
        mapping = {}
        available_qubits = list(env.chip_model.keys())
        
        # 按质量排序（这里用T1时间作为质量指标）
        available_qubits.sort(key=lambda qid: env.chip_model[qid].t1, reverse=True)
        
        # 为每个逻辑量子比特分配物理量子比特
        for logical_idx in range(current_task.num_qubits):
            if logical_idx < len(available_qubits):
                mapping[logical_idx] = available_qubits[logical_idx]
        
        # 计算开始时间
        start_time = 0
        if mapping:
            start_time = max(env.chip_model[pid].get_next_available_time() 
                           for pid in mapping.values())
        
        # 执行调度
        action = {
            "task_id": step,
            "mapping": mapping,
            "start_time": start_time
        }
        
        obs, reward, terminated, _, _ = env.step(action)
        print(f"  Mapped to: {mapping}")
        print(f"  Start time: {start_time:.2f} μs")
        print(f"  Reward: {reward:.4f}")
        
        if terminated:
            break
    
    # 5. 可视化最终调度状态
    print("\n5. Generating final scheduling state visualization...")
    scheduling_state_path = os.path.join(results_dir, "scheduling_state.png")
    env.visualize_scheduling_state(scheduling_state_path)
    
    # 6. 打印最终调度方案
    print("\n=== Final Schedule Plan ===")
    if env.schedule_plan:
        makespan = max(t['end_time'] for t in env.schedule_plan)
        print(f"Makespan: {makespan:.2f} μs")
        
        for i, entry in enumerate(env.schedule_plan):
            print(f"Task {entry['task_id']}: "
                  f"[{entry['start_time']:.2f}, {entry['end_time']:.2f}] μs")
            print(f"  Mapping: {entry['mapping']}")
    
    print(f"\n=== Demo Completed ===")
    print(f"All visualization files saved to: {results_dir}")
    print("Files generated:")
    print(f"  - {chip_overview_path}")
    print(f"  - {connectivity_path}")
    print(f"  - {stats_path}")
    print(f"  - {scheduling_state_path}")
    
    return env


def analyze_chip_quality(env):
    """分析芯片质量分布"""
    print("\n=== Chip Quality Analysis ===")
    
    # 收集所有量子比特的属性
    t1_values = [q.t1 for q in env.chip_model.values()]
    t2_values = [q.t2 for q in env.chip_model.values()]
    f1q_values = [q.fidelity_1q for q in env.chip_model.values()]
    fro_values = [q.fidelity_readout for q in env.chip_model.values()]
    
    import numpy as np
    
    print(f"T1 coherence time: {np.mean(t1_values):.2f} ± {np.std(t1_values):.2f} μs")
    print(f"T2 coherence time: {np.mean(t2_values):.2f} ± {np.std(t2_values):.2f} μs")
    print(f"Single-qubit fidelity: {np.mean(f1q_values):.4f} ± {np.std(f1q_values):.4f}")
    print(f"Readout fidelity: {np.mean(fro_values):.4f} ± {np.std(fro_values):.4f}")
    
    # 找出质量最好和最差的量子比特
    best_qubit_id = max(env.chip_model.keys(), 
                       key=lambda qid: env.chip_model[qid].t1 * env.chip_model[qid].fidelity_1q)
    worst_qubit_id = min(env.chip_model.keys(), 
                        key=lambda qid: env.chip_model[qid].t1 * env.chip_model[qid].fidelity_1q)
    
    print(f"\nBest qubit: {best_qubit_id}")
    best_qubit = env.chip_model[best_qubit_id]
    print(f"  T1: {best_qubit.t1:.2f} μs, T2: {best_qubit.t2:.2f} μs")
    print(f"  Fidelity 1Q: {best_qubit.fidelity_1q:.4f}, Readout: {best_qubit.fidelity_readout:.4f}")
    
    print(f"\nWorst qubit: {worst_qubit_id}")
    worst_qubit = env.chip_model[worst_qubit_id]
    print(f"  T1: {worst_qubit.t1:.2f} μs, T2: {worst_qubit.t2:.2f} μs")
    print(f"  Fidelity 1Q: {worst_qubit.fidelity_1q:.4f}, Readout: {worst_qubit.fidelity_readout:.4f}")


if __name__ == "__main__":
    # 运行主演示
    env = main()
    
    # 运行质量分析
    analyze_chip_quality(env)
    
    print("\n" + "="*50)
    print("Visualization demo completed successfully!")
    print("Check the 'results/chip_visualization' directory for all generated files.")
