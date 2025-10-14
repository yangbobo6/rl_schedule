"""
量子芯片可视化工具
提供多种可视化方式来展示量子芯片的物理属性、连接性和状态信息
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional
import os

from matplotlib import cm
from matplotlib.colors import Normalize


class QuantumChipVisualizer:
    """量子芯片可视化器，支持多种可视化方式"""
    
    def __init__(self, chip_model: Dict[Tuple[int, int], 'PhysicalQubit'], chip_size: Tuple[int, int]):
        self.chip_model = chip_model
        self.chip_size = chip_size
        self.rows, self.cols = chip_size
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def visualize_chip_overview(self, save_path: Optional[str] = None, show_values: bool = True):
        """
        创建芯片总览图，包含所有关键属性的子图
        这是业界最常用的综合可视化方式
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Quantum Chip Overview ({self.rows}×{self.cols})', fontsize=16, fontweight='bold')
        
        # 1. T1 coherence time heatmap
        self._plot_property_heatmap(axes[0, 0], 't1', 'T1 Coherence Time (μs)', 'Blues')
        
        # 2. T2 coherence time heatmap  
        self._plot_property_heatmap(axes[0, 1], 't2', 'T2 Coherence Time (μs)', 'Greens')
        
        # 3. Single-qubit fidelity heatmap
        self._plot_property_heatmap(axes[0, 2], 'fidelity_1q', 'Single-Qubit Fidelity', 'Reds')
        
        # 4. Readout fidelity heatmap
        self._plot_property_heatmap(axes[1, 0], 'fidelity_readout', 'Readout Fidelity', 'Purples')
        
        # 5. Connectivity graph
        self._plot_connectivity_graph(axes[1, 1])
        
        # 6. Quality factor overview
        self._plot_quality_overview(axes[1, 2], show_values)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chip overview saved to: {save_path}")
        
        plt.show()
        
    def _plot_property_heatmap(self, ax, property_name: str, title: str, colormap: str):
        """绘制属性热力图"""
        # 创建数据矩阵
        data = np.zeros((self.rows, self.cols))
        
        for (r, c), qubit in self.chip_model.items():
            if property_name == 't1':
                data[r, c] = qubit.t1
            elif property_name == 't2':
                data[r, c] = qubit.t2
            elif property_name == 'fidelity_1q':
                data[r, c] = qubit.fidelity_1q
            elif property_name == 'fidelity_readout':
                data[r, c] = qubit.fidelity_readout
                
        # 绘制热力图
        im = ax.imshow(data, cmap=colormap, aspect='equal')
        ax.set_title(title, fontweight='bold')
        
        # 添加数值标注
        for r in range(self.rows):
            for c in range(self.cols):
                text = ax.text(c, r, f'{data[r, c]:.3f}', 
                             ha="center", va="center", color="white", fontsize=8)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 设置坐标轴
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
    def _plot_connectivity_graph(self, ax):
        """绘制连接性图"""
        G = nx.Graph()
        
        # 添加节点和边
        pos = {}
        for (r, c), qubit in self.chip_model.items():
            node_id = f"({r},{c})"
            G.add_node(node_id)
            pos[node_id] = (c, self.rows - 1 - r)  # 翻转y轴以匹配矩阵显示
            
            # 添加连接边
            for neighbor_id in qubit.connectivity:
                neighbor_node = f"({neighbor_id[0]},{neighbor_id[1]})"
                if neighbor_node not in G.nodes():
                    continue
                # 边的权重为两量子比特连接的平均保真度
                weight = qubit.connectivity[neighbor_id]['fidelity_2q']
                G.add_edge(node_id, neighbor_node, weight=weight)
        
        # 绘制图
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        
        ax.set_title('Qubit Connectivity Graph', fontweight='bold')
        ax.set_aspect('equal')
        
    def _plot_quality_overview(self, ax, show_values: bool = True):
        """绘制质量因子总览"""
        # 计算每个量子比特的综合质量分数
        quality_scores = []
        positions = []
        
        for (r, c), qubit in self.chip_model.items():
            # 综合质量分数 (归一化)
            t1_norm = qubit.t1 / 120  # 假设最大T1为120μs
            t2_norm = qubit.t2 / 110  # 假设最大T2为110μs  
            f1q_norm = qubit.fidelity_1q
            fro_norm = qubit.fidelity_readout
            
            quality = (t1_norm + t2_norm + f1q_norm + fro_norm) / 4
            quality_scores.append(quality)
            positions.append((c, self.rows - 1 - r))
        
        # 创建散点图
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        scatter = ax.scatter(x_coords, y_coords, c=quality_scores, 
                           cmap='RdYlGn', s=200, alpha=0.8, edgecolors='black')
        
        # 添加数值标注
        if show_values:
            for i, ((r, c), score) in enumerate(zip(self.chip_model.keys(), quality_scores)):
                ax.text(c, self.rows - 1 - r, f'{score:.2f}', 
                       ha="center", va="center", fontsize=8, fontweight='bold')
        
        ax.set_title('Overall Quality Score', fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.grid(True, alpha=0.3)
        
        # 添加colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.8, label='Quality Score')
        
    def visualize_scheduling_state(self, schedule_plan: List[Dict], current_time: float = 0, 
                                 save_path: Optional[str] = None):
        """
        可视化当前调度状态，显示哪些量子比特正在被使用
        这对于调度算法的调试非常有用
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Quantum Chip Scheduling State (t={current_time:.2f}μs)', 
                    fontsize=14, fontweight='bold')
        
        # 左图：当前占用状态
        self._plot_occupancy_state(ax1, schedule_plan, current_time)
        
        # 右图：下次可用时间
        self._plot_availability_heatmap(ax2)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scheduling state saved to: {save_path}")
            
        plt.show()
        
    def _plot_occupancy_state(self, ax, schedule_plan: List[Dict], current_time: float):
        """绘制当前占用状态"""
        # 创建占用状态矩阵
        occupancy = np.zeros((self.rows, self.cols))
        task_labels = {}
        
        for task in schedule_plan:
            if task['start_time'] <= current_time <= task['end_time']:
                for logical_qubit, physical_qubit in task['mapping'].items():
                    r, c = physical_qubit
                    occupancy[r, c] = 1
                    task_labels[physical_qubit] = task['task_id']
        
        # 绘制热力图
        im = ax.imshow(occupancy, cmap='RdYlBu_r', aspect='equal', vmin=0, vmax=1)
        
        # 添加任务标签
        for (r, c) in self.chip_model.keys():
            if occupancy[r, c] == 1:
                task_id = task_labels.get((r, c), '?')
                ax.text(c, r, f'T{task_id}', ha="center", va="center", 
                       color="white", fontweight='bold', fontsize=10)
            else:
                ax.text(c, r, 'Free', ha="center", va="center", 
                       color="gray", fontsize=8)
        
        ax.set_title('Current Qubit Occupancy', fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        
    def _plot_availability_heatmap(self, ax):
        """绘制下次可用时间热力图"""
        availability_data = np.zeros((self.rows, self.cols))
        
        for (r, c), qubit in self.chip_model.items():
            availability_data[r, c] = qubit.get_next_available_time()
        
        im = ax.imshow(availability_data, cmap='YlOrRd', aspect='equal')
        
        # 添加数值标注
        for r in range(self.rows):
            for c in range(self.cols):
                ax.text(c, r, f'{availability_data[r, c]:.1f}', 
                       ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title('Next Available Time (μs)', fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        
        plt.colorbar(im, ax=ax, shrink=0.8)

    def visualize_connectivity_with_edge_labels(self, save_path: Optional[str] = None, positions: dict = None):
        """
        在芯片图上绘制节点与边，并在边上标注双比特门保真度。
        - self.chip_model: dict, key=qubit_id, value=qubit object with .connectivity mapping neighbor->conn_info
        - positions: optional dict {qubit_id: (x,y)} 如果传入则使用，否则尝试按接近方形网格自动布局
        """
        # 准备数据
        n_qubits = len(self.chip_model)
        qubit_list = sorted(self.chip_model.keys())
        qubit_to_idx = {qid: i for i, qid in enumerate(qubit_list)}

        # 收集边和保真度
        edges = []
        fidelities = []
        for qid, qubit in self.chip_model.items():
            i = qubit_to_idx[qid]
            for neighbor_id, conn_info in qubit.connectivity.items():
                # 避免重复添加无向边 —— 只添加 i < j 的边（如果是有向网络，根据情况调整）
                if neighbor_id not in qubit_to_idx:
                    continue
                j = qubit_to_idx[neighbor_id]
                if i < j:
                    fid = conn_info.get('fidelity_2q', 0.0)
                    edges.append((qid, neighbor_id))
                    fidelities.append(float(fid))

        # 如果没有边，直接提示并画出节点
        if len(edges) == 0:
            print("No connectivity edges found in chip_model.")

        # 节点位置：如果没有传入 positions，尝试近似方形网格布局
        if positions is None:
            # 尝试取接近平方的行列数
            rows = int(math.sqrt(n_qubits))
            if rows * rows < n_qubits:
                rows = rows
            cols = math.ceil(n_qubits / rows)
            # 放置节点：按行优先（row, col），y 轴上反向以匹配常见芯片图视觉（0 在底部）
            positions = {}
            for idx, qid in enumerate(qubit_list):
                r = idx // cols
                c = idx % cols
                # x = column, y = rows - 1 - row 使得索引 0 在左下角
                positions[qid] = (c, rows - 1 - r)

        # 构建 NetworkX 图用于绘制
        G = nx.Graph()
        G.add_nodes_from(qubit_list)
        G.add_edges_from(edges)

        # 设置 colormap 范围（忽略 0 的无效值）
        if len(fidelities) > 0:
            fid_array = np.array(fidelities)
            valid_mask = fid_array > 0
            if valid_mask.any():
                vmin = float(np.min(fid_array[valid_mask]))
                vmax = float(np.max(fid_array[valid_mask]))
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0

        # 防止 vmin==vmax 导致归一化出错
        if vmin == vmax:
            vmin = max(0.0, vmin - 0.01)
            vmax = min(1.0, vmax + 0.01)

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('RdYlGn')

        # 将边映射到颜色与宽度
        edge_colors = []
        edge_widths = []
        for fid in fidelities:
            # 若 fid 为 0（表示无连接或者未设置），设置为灰色/细线
            if fid <= 0:
                edge_colors.append((0.6, 0.6, 0.6, 0.4))
                edge_widths.append(0.5)
            else:
                edge_colors.append(cmap(norm(fid)))
                # 宽度按保真度线性映射，保证可视差异
                edge_widths.append(0.5 + 4.0 * ((fid - vmin) / (vmax - vmin)))

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Quantum Chip — Two-Qubit Fidelity on Edges', fontsize=14, fontweight='bold')

        # 画节点
        nx.draw_networkx_nodes(G, positions, node_size=350, node_color='lightgrey', ax=ax, edgecolors='k')
        # 节点标签为 qubit 索引或 id（这里用索引方便对应矩阵）
        idx_labels = {qid: str(qubit_to_idx[qid]) for qid in qubit_list}
        nx.draw_networkx_labels(G, positions, labels=idx_labels, font_size=9, ax=ax)

        # 画边
        nx.draw_networkx_edges(G, positions, edgelist=edges, edge_color=edge_colors, width=edge_widths, ax=ax)

        # 在每条边上标注保真度（中点处）
        for (u, v), fid in zip(edges, fidelities):
            (xu, yu) = positions[u]
            (xv, yv) = positions[v]
            xm = (xu + xv) / 2.0
            ym = (yu + yv) / 2.0
            if fid > 0:
                txt = f"{fid:.3f}"
                # 若边比较倾斜，稍微偏移 label，避免与边重叠
                dx = (yv - yu) * 0.08
                dy = (xu - xv) * 0.02
                ax.text(xm + dx, ym + dy, txt, fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
            else:
                # 可选择不标注无效边
                pass

        # colorbar：用一个 ScalarMappable 来生成色条
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 仅为 colorbar 使用
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Two-Qubit Gate Fidelity')

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chip connectivity with fidelities saved to: {save_path}")

        plt.show()
        
    def export_chip_stats(self, save_path: Optional[str] = None) -> Dict:
        """
        导出芯片统计信息，用于性能分析和比较
        """
        stats = {
            'chip_size': self.chip_size,
            'total_qubits': len(self.chip_model),
            'avg_t1': np.mean([q.t1 for q in self.chip_model.values()]),
            'avg_t2': np.mean([q.t2 for q in self.chip_model.values()]),
            'avg_fidelity_1q': np.mean([q.fidelity_1q for q in self.chip_model.values()]),
            'avg_fidelity_readout': np.mean([q.fidelity_readout for q in self.chip_model.values()]),
            'total_connections': sum(len(q.connectivity) for q in self.chip_model.values()) // 2,
            'avg_connectivity': np.mean([len(q.connectivity) for q in self.chip_model.values()]),
        }
        
        # 计算两量子比特门的平均保真度
        two_qubit_fidelities = []
        for qubit in self.chip_model.values():
            for conn_info in qubit.connectivity.values():
                two_qubit_fidelities.append(conn_info['fidelity_2q'])
        
        if two_qubit_fidelities:
            stats['avg_fidelity_2q'] = np.mean(two_qubit_fidelities)
            stats['min_fidelity_2q'] = np.min(two_qubit_fidelities)
            stats['max_fidelity_2q'] = np.max(two_qubit_fidelities)
        
        if save_path:
            import json
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Chip statistics saved to: {save_path}")
        
        return stats


def demo_visualization(chip_model, chip_size):
    """演示所有可视化功能"""
    visualizer = QuantumChipVisualizer(chip_model, chip_size)
    
    print("=== Quantum Chip Visualization Demo ===")
    
    # 1. 芯片总览
    print("1. Generating chip overview...")
    visualizer.visualize_chip_overview("results/chip_overview.png")
    
    # 2. 连接矩阵
    print("2. Generating connectivity matrix...")
    visualizer.visualize_connectivity_matrix("results/connectivity_matrix.png")
    
    # 3. 导出统计信息
    print("3. Exporting chip statistics...")
    stats = visualizer.export_chip_stats("results/chip_stats.json")
    
    print("Statistics Summary:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nVisualization demo completed!")
    return visualizer
