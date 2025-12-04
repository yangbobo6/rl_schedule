from typing import Dict, List

import matplotlib

# 即使关闭了PyCharm的工具窗口，使用'Agg'后端也是一个好习惯
# 'Agg'是一个非交互式后端，专门用于生成图像文件，可以避免任何GUI问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import matplotlib.patches as patches


def plot_schedule(schedule_plan: list,
                  chip_size: tuple,
                  baseline_makespan: float = None,
                  save_path: str = None):
    """
    使用Matplotlib绘制2D甘特图来可视化调度方案。
    如果提供了save_path，则将图像保存到文件；否则，显示图像。

    Args:
        schedule_plan (list): 环境生成的完整调度方案。
        chip_size (tuple): 芯片尺寸, e.g., (5, 5).
        baseline_makespan (float, optional): 基线的Makespan，用于对比。
        save_path (str, optional): 保存图像的文件路径。例如, "results/run_1/episode_50.png".
    """
    if not schedule_plan:
        print("Cannot plot an empty schedule.")
        return

    fig, ax = plt.subplots(figsize=(18, 10))  # 增大图像尺寸以容纳更多信息

    num_qubits = chip_size[0] * chip_size[1]
    ax.set_ylim(-0.5, num_qubits - 0.5)
    ax.set_yticks(np.arange(num_qubits))
    ax.set_yticklabels([f"Q({i // chip_size[1]},{i % chip_size[1]})" for i in range(num_qubits)])
    ax.set_xlabel("Time (us)", fontsize=12)
    ax.set_ylabel("Physical Qubit", fontsize=12)
    ax.set_title("Quantum Circuit Schedule Gantt Chart", fontsize=16)

    # 为每个任务ID分配一个颜色
    task_ids = sorted(list(set(item['task_id'] for item in schedule_plan)))
    # 使用更鲜艳的颜色映射
    colors = plt.cm.get_cmap('viridis', len(task_ids)) if len(task_ids) > 0 else plt.cm.get_cmap('viridis')
    task_colors = {tid: colors(i / len(task_ids)) for i, tid in enumerate(task_ids)} if task_ids else {}

    # 物理比特ID到Y轴位置的映射
    qubit_map = {(r, c): r * chip_size[1] + c for r in range(chip_size[0]) for c in range(chip_size[1])}

    for item in schedule_plan:
        task_id = item['task_id']
        start_time = item['start_time']
        duration = item['end_time'] - start_time

        for physical_id in item['mapping'].values():
            y_pos = qubit_map[physical_id]

            # 绘制矩形代表任务
            rect = Rectangle(
                (start_time, y_pos - 0.4), duration, 0.8,  # 调整高度和位置以获得更好的视觉效果
                facecolor=task_colors.get(task_id, 'gray'), edgecolor='black', alpha=0.8, linewidth=0.5
            )
            ax.add_patch(rect)

            # 在矩形中间添加任务ID文本
            if duration > 0:  # 避免在零长度矩形上写字
                ax.text(start_time + duration / 2, y_pos, f"T{task_id}",
                        ha='center', va='center', color='white', weight='bold', fontsize=8)

    final_makespan = max(item['end_time'] for item in schedule_plan) if schedule_plan else 0
    ax.set_xlim(0, final_makespan * 1.1)

    # 画一条红线表示最终的Makespan
    ax.axvline(x=final_makespan, color='r', linestyle='--', linewidth=2, label=f'RL Makespan: {final_makespan:.2f} us')

    # 如果有基线，也画出来
    if baseline_makespan:
        ax.axvline(x=baseline_makespan, color='b', linestyle=':', linewidth=2,
                   label=f'Baseline Makespan: {baseline_makespan:.2f} us')

    ax.legend(fontsize=10)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()

    if save_path:
        # 确保目录存在
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(save_path, dpi=150)  # 保存文件，dpi提高清晰度
        print(f"Schedule plot saved to {save_path}")
    else:
        plt.show()  # 如果没有提供路径，则显示图像

    plt.close(fig)  # 关闭图形对象，释放内存，这一点非常重要！



def plot_comparison(rl_plan: list, qumc_plan: list, chip_size: tuple, save_path: str = None):
    """
    在两个子图上并排（上下）绘制RL和QuMC的调度方案以进行对比。
    
    Args:
        rl_plan (list): RL调度器生成的调度计划。
        qumc_plan (list): QuMC启发式生成的调度计划。
        chip_size (tuple): 芯片尺寸, e.g., (6, 6).
        save_path (str, optional): 保存对比图像的文件路径。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    fig.suptitle('Scheduler Comparison: Reinforcement Learning vs. QuMC Heuristic', fontsize=20, y=0.98)

    plans = [rl_plan, qumc_plan]
    axes = [ax1, ax2]
    titles = ["Reinforcement Learning Scheduler", "QuMC-style Heuristic"]
    colors = ['r', 'b']
    
    max_makespan = 0
    # 先计算最大的makespan，以统一坐标轴
    for plan in plans:
        if plan:
            current_makespan = max(item['end_time'] for item in plan)
            if current_makespan > max_makespan:
                max_makespan = current_makespan
    
    # 循环绘制两个子图
    for i, plan in enumerate(plans):
        ax = axes[i]
        title = titles[i]
        
        num_qubits = chip_size[0] * chip_size[1]
        ax.set_ylim(-0.5, num_qubits - 0.5)
        ax.set_yticks(np.arange(num_qubits))
        ax.set_yticklabels([f"Q({r},{c})" for r in range(chip_size[0]) for c in range(chip_size[1])])
        ax.set_ylabel("Physical Qubit")
        ax.set_title(title, fontsize=14)
        ax.grid(axis='x', linestyle=':', alpha=0.7)

        if not plan: 
            ax.text(0.5, 0.5, 'No schedule generated.', ha='center', va='center', transform=ax.transAxes)
            continue
            
        task_ids = sorted(list(set(item['task_id'] for item in plan)))
        cmap = plt.cm.get_cmap('viridis', len(task_ids)) if task_ids else plt.cm.get_cmap('viridis')
        task_colors = {tid: cmap(j / len(task_ids)) for j, tid in enumerate(task_ids)} if task_ids else {}
        
        qubit_map = {(r, c): r * chip_size[1] + c for r in range(chip_size[0]) for c in range(chip_size[1])}

        for item in plan:
            task_id = item['task_id']
            start_time = item['start_time']
            duration = item['end_time'] - start_time
            
            if 'mapping' in item and item['mapping']:
                for logical_qubit_id, physical_id in item['mapping'].items():
                    y_pos = qubit_map.get(physical_id)
                    if y_pos is not None:
                        rect = Rectangle(
                            (start_time, y_pos - 0.4), duration, 0.8,
                            facecolor=task_colors.get(task_id, 'gray'), edgecolor='black', alpha=0.8, linewidth=0.5
                        )
                        ax.add_patch(rect)
                        if duration > 1000: # 只在较长的条上写字，避免拥挤
                            ax.text(start_time + duration / 2, y_pos, f"T{task_id}", 
                                    ha='center', va='center', color='white', weight='bold', fontsize=7)

        current_makespan = max(item['end_time'] for item in plan) if plan else 0
        ax.axvline(x=current_makespan, color=colors[i], linestyle='--', linewidth=2, label=f'Makespan: {current_makespan:.2f} us')
        ax.legend()
        ax.set_xlim(0, max_makespan * 1.1)

    ax2.set_xlabel("Time (us)", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局以适应总标题
    
    if save_path:
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=200)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close(fig) # 关闭图形对象，释放内存


def plot_all_schedules(plans: Dict[str, List[Dict]],
                       chip_size: tuple,
                       save_path: str = None):
    """
    在一张大图上，为多个调度器方案绘制各自的甘特图子图。

    Args:
        plans (Dict[str, List[Dict]]): 一个字典，键是调度器名称，值是对应的schedule_plan。
        chip_size (tuple): 芯片尺寸, e.g., (6, 6).
        save_path (str, optional): 保存图像的文件路径。
    """
    num_schedulers = len(plans)
    if num_schedulers == 0:
        print("No schedules to plot.")
        return

    # 创建 (N, 1) 排列的子图，N是调度器的数量
    fig, axes = plt.subplots(num_schedulers, 1, figsize=(20, 7 * num_schedulers), sharex=True)
    fig.suptitle('Comparison of Different Scheduling Algorithms', fontsize=22, y=0.99)

    # 如果只有一个调度器，确保axes是一个列表
    if num_schedulers == 1:
        axes = [axes]

    # 找到所有方案中的最大makespan，以统一X轴
    max_makespan = 0
    for plan in plans.values():
        if plan:
            current_max = max(item['end_time'] for item in plan)
            if current_max > max_makespan:
                max_makespan = current_max

    # 提取所有任务ID，以创建统一的颜色映射
    all_task_ids = sorted(list(set(item['task_id'] for plan in plans.values() for item in plan)))
    cmap = plt.cm.get_cmap('viridis', len(all_task_ids)) if all_task_ids else plt.cm.get_cmap('viridis')
    task_colors = {tid: cmap(i / len(all_task_ids)) for i, tid in enumerate(all_task_ids)} if all_task_ids else {}

    num_qubits = chip_size[0] * chip_size[1]
    qubit_map = {(r, c): r * chip_size[1] + c for r in range(chip_size[0]) for c in range(chip_size[1])}

    # 遍历每个调度器和它对应的子图
    for ax, (name, plan) in zip(axes, plans.items()):

        ax.set_ylim(-0.5, num_qubits - 0.5)
        ax.set_yticks(np.arange(num_qubits))
        ax.set_yticklabels([f"Q({r},{c})" for r in range(chip_size[0]) for c in range(chip_size[1])])
        ax.set_ylabel("Physical Qubit")
        ax.set_title(name, fontsize=16)
        ax.grid(axis='x', linestyle=':', alpha=0.7)

        if not plan:
            ax.text(0.5, 0.5, "No schedule generated.", ha='center', va='center', transform=ax.transAxes)
            continue

        for item in plan:
            task_id = item['task_id']
            start_time = item['start_time']
            duration = item['end_time'] - start_time

            if 'mapping' in item and item['mapping']:
                for physical_id in item['mapping'].values():
                    y_pos = qubit_map.get(physical_id)
                    if y_pos is not None:
                        rect = Rectangle(
                            (start_time, y_pos - 0.4), duration, 0.8,
                            facecolor=task_colors.get(task_id, 'gray'), edgecolor='black', alpha=0.8, linewidth=0.5
                        )
                        ax.add_patch(rect)
                        if duration > 100:  # 只在较长的条上显示文本，避免拥挤
                            ax.text(start_time + duration / 2, y_pos, f"T{task_id}",
                                    ha='center', va='center', color='white', weight='bold', fontsize=7)

        current_makespan = max(item['end_time'] for item in plan) if plan else 0
        ax.axvline(x=current_makespan, color='r', linestyle='--', linewidth=2,
                   label=f'Makespan: {current_makespan:.2f} us')
        ax.legend()

    # 为所有子图设置统一的X轴范围和标签
    for ax in axes:
        ax.set_xlim(0, max_makespan * 1.05)
    axes[-1].set_xlabel("Time (us)", fontsize=14)  # 只在最下面的图显示X轴标签

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 调整布局以适应大标题

    if save_path:
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"All schedules plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_chip_snapshot(schedule_plan, chip_size, snapshot_time, save_path=None):
    """
    绘制特定时间点 (snapshot_time) 的芯片任务分布切面图。
    """
    rows, cols = chip_size
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. 绘制芯片网格背景
    for r in range(rows):
        for c in range(cols):
            # 绘制物理比特圆圈
            circle = patches.Circle((c, rows - 1 - r), 0.3, edgecolor='gray', facecolor='white', linewidth=2,
                                    zorder=1)
            ax.add_patch(circle)
            # 绘制连接线 (简化为网格线)
            if c < cols - 1:
                ax.plot([c, c + 1], [rows - 1 - r, rows - 1 - r], color='gray', linewidth=1, zorder=0)
            if r < rows - 1:
                ax.plot([c, c], [rows - 1 - r, rows - 1 - (r + 1)], color='gray', linewidth=1, zorder=0)

            # 标注比特坐标
            ax.text(c, rows - 1 - r, f"({r},{c})", ha='center', va='center', fontsize=8, color='black', zorder=2)

    # 2. 找出在 snapshot_time 正在运行的任务
    active_tasks = []
    for item in schedule_plan:
        if item['start_time'] <= snapshot_time < item['end_time']:
            active_tasks.append(item)

    # 3. 绘制活跃任务
    # 为每个任务分配颜色
    task_ids = sorted(list(set(item['task_id'] for item in schedule_plan)))
    cmap = plt.cm.get_cmap('viridis', len(task_ids)) if task_ids else None

    print(f"\n--- Snapshot at {snapshot_time} us ---")
    print(f"Active tasks: {[t['task_id'] for t in active_tasks]}")

    for item in active_tasks:
        task_id = item['task_id']
        color = cmap(task_ids.index(task_id))

        # 获取该任务占用的物理比特
        mapped_qubits = item['mapping'].values()  # [(r1, c1), (r2, c2), ...]

        # 在对应的物理比特上绘制彩色圆圈
        for (r, c) in mapped_qubits:
            # 注意坐标变换：矩阵行r对应Y轴从上到下，所以Y = rows - 1 - r
            circle = patches.Circle((c, rows - 1 - r), 0.25, color=color, zorder=3)
            ax.add_patch(circle)
            # 标注任务ID
            ax.text(c, rows - 1 - r, f"T{task_id}", ha='center', va='center', fontsize=10, color='white',
                    weight='bold', zorder=4)

    # 设置图形属性
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Chip Snapshot at T = {snapshot_time} us", fontsize=16)

    # 添加图例 (可选，显示所有活跃任务)
    if active_tasks:
        handles = [patches.Patch(color=cmap(task_ids.index(t['task_id'])), label=f"Task {t['task_id']}") for t in
                   active_tasks]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Snapshot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
