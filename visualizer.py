import matplotlib

# 即使关闭了PyCharm的工具窗口，使用'Agg'后端也是一个好习惯
# 'Agg'是一个非交互式后端，专门用于生成图像文件，可以避免任何GUI问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os


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
    在两个子图上并排绘制RL和QuMC的调度方案以进行对比。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), sharex=False)
    fig.suptitle('Scheduler Comparison: Reinforcement Learning vs. QuMC Heuristic', fontsize=20)

    plans = [rl_plan, qumc_plan]
    axes = [ax1, ax2]
    titles = ["RL Scheduler", "QuMC-style Heuristic"]
    colors = ['r', 'b']

    max_makespan = 0

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

        if not plan: continue

        task_ids = sorted(list(set(item['task_id'] for item in plan)))
        cmap = plt.cm.get_cmap('viridis', len(task_ids)) if task_ids else plt.cm.get_cmap('viridis')
        task_colors = {tid: cmap(j / len(task_ids)) for j, tid in enumerate(task_ids)} if task_ids else {}

        qubit_map = {(r, c): r * chip_size[1] + c for r in range(chip_size[0]) for c in range(chip_size[1])}

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
                        if duration > 0:
                            ax.text(start_time + duration / 2, y_pos, f"T{task_id}",
                                    ha='center', va='center', color='white', weight='bold', fontsize=8)

        current_makespan = max(item['end_time'] for item in plan) if plan else 0
        if current_makespan > max_makespan:
            max_makespan = current_makespan

        ax.axvline(x=current_makespan, color=colors[i], linestyle='--', linewidth=2,
                   label=f'Makespan: {current_makespan:.2f} us')
        ax.legend()

    ax1.set_xlim(0, max_makespan * 1.1)
    ax2.set_xlim(0, max_makespan * 1.1)
    ax2.set_xlabel("Time (us)", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=200)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
