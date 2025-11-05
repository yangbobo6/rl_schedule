from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm

# --- 1. 导入自定义模块 ---
from environment import QuantumSchedulingEnv
from models.actor_critic import CNNActorCritic, TransformerActorCritic
from models.gnn_encoder import GNNEncoder, networkx_to_pyg_data
from models.quantum_task import QuantumTask  # 确保__init__.py工作正常
from torch.utils.tensorboard import SummaryWriter # 导入TensorBoard的Writer
import time
import os # 导入os模块
from visualizer import plot_schedule # 导入我们的绘图函数

# --- 2. 定义超参数 ---
class Hyperparameters:
    # --- 训练过程参数 ---
    MAX_EPISODES = 50000  # 增加训练总轮数

    # --- 环境参数 ---
    CHIP_SIZE = (6, 6)  # 挑战更大的芯片
    NUM_TASKS = 20  # 挑战更多的任务
    MAX_QUBITS_PER_TASK = 10  # 相应增加

    # --- PPO 训练参数 ---
    LEARNING_RATE = 1e-4  # Transformer的稳定学习率
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    PPO_EPSILON = 0.2
    CRITIC_DISCOUNT = 1  # 保持较高的Critic权重
    ENTROPY_BETA = 0.01  # 可以从一个稍小的值开始，防止在长训练中过早停止探索
    PPO_EPOCHS = 10  # 保持较高的PPO Epochs
    ROLLOUT_LENGTH = 16384  # 充分利用GPU和内存，收集高质量数据
    MINI_BATCH_SIZE = 256  # 使用更大的批次以获得更稳定的梯度

    # --- GNN参数 ---
    GNN_NODE_DIM = 1
    GNN_HIDDEN_DIM = 32
    GNN_OUTPUT_DIM = 16

    # --- Transformer模型参数 ---
    D_MODEL = 128
    N_HEAD = 4
    NUM_ENCODER_LAYERS = 3

    # 奖励权重
    REWARD_WEIGHTS = {
        "compaction": 2.5, # 时间压缩
        "swap": 1.5,       # swap次数
        "fidelity": 0.5,   # 保真度
        "crosstalk": 0.7,  # 串扰
        "matching": 0.6    # 启发式奖励权重
    }


# --- 3. 定义Actor-Critic网络 ---
# 这是一个简化的MLP版本，作为起点。之后可以替换为Transformer。
class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space_dim):
        super(ActorCritic, self).__init__()

        # 简化: 将字典观察空间展平
        # 实际中，Transformer能更好地处理结构化数据
        qubit_embed_dim = np.prod(obs_space["qubit_embeddings"].shape)
        task_embed_dim = np.prod(obs_space["current_task_embedding"].shape)
        placement_mask_dim = np.prod(obs_space["placement_mask"].shape)
        logic_context_dim = np.prod(obs_space["logical_qubit_context"].shape)

        input_dim = qubit_embed_dim + task_embed_dim + placement_mask_dim + logic_context_dim

        # 共享层
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor头 (输出动作概率)
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_dim)  # 输出每个物理比特的logits
        )

        # Critic头 (输出状态价值)
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出一个标量值
        )

    def forward(self, obs):
        # 将字典观察展平并拼接
        flat_obs = torch.cat([
            obs["qubit_embeddings"].flatten(start_dim=1),
            obs["current_task_embedding"].flatten(start_dim=1),
            obs["placement_mask"].flatten(start_dim=1).float(),
            obs["logical_qubit_context"].flatten(start_dim=1)
        ], dim=1)

        shared_features = self.shared_net(flat_obs)

        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)

        return action_logits, state_value


from models.selector import SiameseGNNSelector

# --- 4. 主函数 ---
def main():

    # --- 初始化 ---
    args = Hyperparameters()
    # --- 创建唯一的运行目录 ---
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"results/{run_timestamp}"
    tensorboard_log_dir = f"{run_dir}/tensorboard_logs"
    plots_dir = f"{run_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    print(f"Results will be saved in: {run_dir}")

    # 创建用于保存模型权重的目录
    checkpoints_dir = f"{run_dir}/checkpoints"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Results will be saved in: {run_dir}")

    # --- TensorBoard 初始化 ---
    run_name = f"QuantumScheduler__{int(time.time())}"
    writer = SummaryWriter(tensorboard_log_dir)
    # 将超参数也写入日志，方便追溯
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    global_step = 0
    # 用于计算平均性能的滑动窗口
    recent_rewards = deque(maxlen=100)
    recent_makespans = deque(maxlen=100)

    # 1. 初始化GNN模型
    gnn_model = GNNEncoder(
        node_feature_dim=args.GNN_NODE_DIM,
        hidden_dim=args.GNN_HIDDEN_DIM,
        output_dim=args.GNN_OUTPUT_DIM
    ).to(device)
    gnn_model.eval()  # 我们只用它来编码，不训练

    # b. GNN选择器 (用于任务选择)
    print("Loading pre-trained GNN Selector...")
    gnn_selector = SiameseGNNSelector(
        chip_node_dim=1, task_node_dim=1,  # 节点特征维度
        hidden_dim=64, embed_dim=32
    ).to(device)
    try:
        gnn_selector.load_state_dict(torch.load("models/gnn_selector_v0.pth", map_location=device))
    except FileNotFoundError:
        print("Warning: GNN Selector model not found. Task selection will be random.")
        gnn_selector = None  # 如果找不到模型，则退化为随机选择
    if gnn_selector:
        gnn_selector.eval()


    gate_times = {
        'u3': 50, 'u': 50, 'p': 30, 'cx': 350, 'id': 30,
        'measure': 1000, 'rz': 30, 'sx': 40, 'x': 40
    }  # 扩展门集以适应更多线路

    env = QuantumSchedulingEnv(
        chip_size=args.CHIP_SIZE,
        num_tasks=args.NUM_TASKS,
        max_qubits_per_task=args.MAX_QUBITS_PER_TASK,
        gnn_model=gnn_model,
        device=device,
        gate_times=gate_times,
        reward_weights=args.REWARD_WEIGHTS
    )

    # --- 保存芯片可视化图表 ---
    print("Generating and saving chip visualizations...")
    chip_viz_dir = f"{run_dir}/chip_visualization"
    os.makedirs(chip_viz_dir, exist_ok=True)

    # 保存芯片总览图
    chip_overview_path = f"{chip_viz_dir}/chip_overview.png"
    env.visualize_chip(chip_overview_path, show_values=True)

    # 保存连接矩阵
    connectivity_path = f"{chip_viz_dir}/connectivity_matrix.png"
    env.visualize_connectivity(connectivity_path)

    # 导出芯片统计信息
    stats_path = f"{chip_viz_dir}/chip_stats.json"
    chip_stats = env.export_chip_stats(stats_path)

    print(f"Chip visualizations saved to: {chip_viz_dir}")
    print(f"Chip stats: T1={chip_stats['avg_t1']:.2f}μs, T2={chip_stats['avg_t2']:.2f}μs, "
          f"Fidelity_1Q={chip_stats['avg_fidelity_1q']:.4f}")

    # 3. 初始化新的Actor-Critic模型
    action_space_dim = env.num_qubits
    # model = CNNActorCritic(env.observation_space, action_space_dim).to(device)
    model = TransformerActorCritic(
        obs_space=env.observation_space,
        action_space_dim=action_space_dim,
        d_model=args.D_MODEL,
        nhead=args.N_HEAD,
        num_encoder_layers=args.NUM_ENCODER_LAYERS
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    # 引入学习率衰减，适用于长时间训练
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                  total_iters=args.MAX_EPISODES)

    # PPO经验缓冲区
    rollout_buffer = {
        "obs": [], "actions": [], "log_probs": [], "rewards": [],
        "dones": [], "values": []
    }

    # --- 训练循环 ---
    global_step = 0
    # 在main函数开始时，计算一个基线
    baseline_makespan = sum(t.estimated_duration for t in env.task_pool.values())
    # 用于计算训练性能滑动平均的窗口
    recent_train_rewards = deque(maxlen=100)
    recent_train_makespans = deque(maxlen=100)
    # 用于跟踪最佳性能的变量
    best_validation_makespan = float('inf')
    episode_pbar = tqdm(range(args.MAX_EPISODES), desc="Training Progress")
    for episode in episode_pbar:
        # =====================================================================
        # === 训练阶段 (Training Phase) ===
        # =====================================================================
        # 告诉环境从'train'池中采样
        initial_obs_dict, info = env.reset(options={'pool': 'train'})

        # --- 宏观循环: 调度一个完整的方案 ---
        # 在Episode开始前，记录一下最终的统计数据
        episode_reward_sum = 0.0  # 用于累加本episode的总奖励
        episode_swaps = []
        episode_fidelities = []
        episode_crosstalks = []

        for task_id in range(env.num_tasks):
            # === 1. 任务选择决策 (由GNN完成) ===
            candidate_tasks = env.get_candidate_tasks()
            if not candidate_tasks: break  # 如果没有可选任务，提前结束

            if gnn_selector:
                chip_graph_data = env.get_chip_state_graph_data()
                task_graph_data_list = [networkx_to_pyg_data(t.interaction_graph) for t in candidate_tasks]

                scores = []
                with torch.no_grad():
                    # 为了高效，我们将所有候选任务打包成一个batch进行预测
                    chip_batch = Batch.from_data_list([chip_graph_data] * len(candidate_tasks)).to(device)
                    task_batch = Batch.from_data_list(task_graph_data_list).to(device)

                    scores = gnn_selector(chip_batch, task_batch).cpu().numpy().flatten()

                # 选择分数最高的任务
                chosen_task_local_index = np.argmax(scores)
                chosen_task = candidate_tasks[chosen_task_local_index]
                task_id = chosen_task.id
            else:
                # 如果没有GNN选择器，则随机选择一个
                chosen_task = np.random.choice(candidate_tasks)
                task_id = chosen_task.id

            # === 2. 任务放置决策 (由RL完成) ===
            current_task = env.task_pool[task_id]
            num_logical_qubits = current_task.num_qubits
            placement_in_progress = {}

            # --- 微观循环: 自回归地放置一个任务 ---
            for i in range(num_logical_qubits):
                global_step += 1

                # 1. 获取观察
                if task_id == 0 and i == 0:
                    # 对于整个episode的第一次决策，使用reset返回的obs
                    obs_dict = initial_obs_dict
                else:
                    # 对于后续决策，正常调用_get_obs
                    obs_dict = env._get_obs(task_id, placement_in_progress)

                # 将Numpy观察转换为Torch张量
                obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in
                              obs_dict.items()}

                # 2. 智能体决策
                with torch.no_grad():
                    # 模型输出的每个动作（这里是每个物理比特）对应的未归一化的分数
                    # Critic 输出的状态价值估计，表示当前状态预计能获得的未来回报
                    logits, value = model(obs_tensor)

                # 构建合法的动作掩码
                # a. 获取已占用的比特掩码
                # tensor([False, False, False, ...]),标记哪些物理比特已经被占用
                placement_mask = torch.tensor(obs_dict["placement_mask"], dtype=torch.bool).to(device)

                # b. 获取连通性约束掩码,基于量子芯片的连通性约束来限制动作选择
                connectivity_mask = torch.ones_like(placement_mask)  # 初始假设所有动作都非法
                current_logical_idx = i
                task_graph = current_task.interaction_graph

                if i == 0:
                    # 第一个比特可以放在任何未占用的地方
                    connectivity_mask = placement_mask.clone()
                else:
                    valid_neighbors = set()
                    # 检查当前逻辑比特 l_i 是否需要与任何已放置的比特 l_j 连接
                    needs_connection = any(
                        task_graph.has_edge(current_logical_idx, j) for j in placement_in_progress.keys())

                    if not needs_connection:
                        # 如果 l_i 与已放置的部分没有连接（例如，一个任务包含两个独立的组件）
                        # 那么它可以放在任何未占用的地方
                        connectivity_mask = placement_mask.clone()
                    else:
                        # 否则，它必须放在已放置邻居的空闲邻居位置
                        for placed_logical_idx, placed_physical_id in placement_in_progress.items():
                            if task_graph.has_edge(current_logical_idx, placed_logical_idx):
                                for neighbor_id in env.chip_model[placed_physical_id].connectivity:
                                    neighbor_idx = env.qubit_id_to_idx[neighbor_id]
                                    if not placement_mask[neighbor_idx]:
                                        valid_neighbors.add(neighbor_idx)

                        if valid_neighbors:
                            indices = torch.tensor(list(valid_neighbors), dtype=torch.long, device=device)
                            connectivity_mask.scatter_(0, indices, 0)  # 将合法位置的掩码设为False
                        else:
                            # 如果所有邻居都被占用了，允许它放在任何未占用的地方，依赖奖励惩罚
                            connectivity_mask = placement_mask.clone()

                # c. 合并掩码
                # 一个动作非法，当且仅当它已被占用 OR 它不满足连通性
                final_mask = placement_mask | connectivity_mask
                logits[0, final_mask] = -1e8  # 将非法动作的logit设为极小值

                probs = Categorical(logits=logits) # PyTorch 的 Categorical 分布对象，把 logits 转换成概率分布
                action = probs.sample()  # 采样一个动作 (物理比特的索引)
                log_prob = probs.log_prob(action)

                # --- 计算微观奖励 ---
                micro_reward = 0
                chosen_physical_id = env.idx_to_qubit_id[action.item()]

                # 检查当前选择是否与已放置的邻居物理相连
                for placed_logic_idx, placed_phys_id in placement_in_progress.items():
                    if task_graph.has_edge(current_logical_idx, placed_logic_idx):
                        if chosen_physical_id in env.chip_model[placed_phys_id].connectivity:
                            micro_reward += 0.1
                        else:
                            # 如果硬约束生效，理论上不会走到这里。
                            # 但如果硬约束被放宽（例如 valid_neighbors 为空时），这个惩罚就至关重要
                            micro_reward -= 0.1

                # 3. 存储经验
                rollout_buffer["obs"].append(obs_dict)
                rollout_buffer["actions"].append(action.cpu().numpy())
                rollout_buffer["log_probs"].append(log_prob.cpu().numpy())
                rollout_buffer["values"].append(value.cpu().numpy())
                rollout_buffer["rewards"].append(micro_reward)
                rollout_buffer["dones"].append(False)

                # 4. 更新进行中的放置
                physical_qubit_id = env.idx_to_qubit_id[action.item()]  # 将action的下表（例如2）转换为qubit_id 例如(2,3)
                placement_in_progress[i] = physical_qubit_id #dict (0:(2,3), 1:(2,4), 2:(3,3), 3:(3,4))

                # 5. 检查是否需要更新网络
                if len(rollout_buffer["actions"]) >= args.ROLLOUT_LENGTH:
                    update_ppo(model, optimizer, rollout_buffer, args, device, value, writer, global_step)
                    # 清空缓冲区
                    for k in rollout_buffer: rollout_buffer[k] = []

            # --- 放置完成后，提交计划给环境 ---
            final_mapping = placement_in_progress
            start_time = 0
            if final_mapping:
                start_time = max(env.chip_model[pid].get_next_available_time() for pid in final_mapping.values())

            full_action = {"task_id": task_id, "mapping": final_mapping, "start_time": start_time}
            _, reward, terminated, _, info = env.step(full_action)
            # 累加指标
            if "num_swaps" in info:
                episode_swaps.append(info["num_swaps"])
                episode_fidelities.append(info["final_fidelity"])
                episode_crosstalks.append(info["crosstalk_score"])
            episode_reward_sum += reward  # 累加到episode总奖励

            # --- 将宏观步骤的奖励分配给微观步骤 ---
            # 这是关键一步：我们将刚刚收到的中间奖励，
            # 平均分配或者全部归功于导致它的最后一个微观动作。
            # 将其归功于最后一个微观动作更简单且常用。  (中间的映射比特没有奖励，只是防止完一个任务才会有奖励)
            if rollout_buffer["rewards"]:
                rollout_buffer["rewards"][-1] += reward

            if terminated:
                break

        # --- Episode 结束 ---
        # 此时episode_reward_sum就是TensorBoard中要记录的值
        # rollout_buffer中已经包含了所有中间奖励和最终奖
        print(f"Episode {episode}: Total Reward = {episode_reward_sum:.4f}")
        writer.add_scalar("charts/episode_reward", episode_reward_sum, episode)

        # --- 计算并记录最终的调度方案指标 ---
        if env.schedule_plan:
            final_makespan = max(t['end_time'] for t in env.schedule_plan)
        else:
            final_makespan = 0

        # --- 计算平均指标并写入TensorBoard ---
        avg_swaps = np.mean(episode_swaps) if episode_swaps else 0
        avg_final_fidelity = np.mean(episode_fidelities) if episode_fidelities else 0
        avg_crosstalk = np.mean(episode_crosstalks) if episode_crosstalks else 0

        final_avg_fidelity = avg_final_fidelity
        final_total_crosstalk = np.sum(episode_crosstalks) if episode_crosstalks else 0

        # 将最新的结果存入滑动窗口
        recent_rewards.append(episode_reward_sum)
        recent_makespans.append(final_makespan)

        # 使用 set_postfix 更新进度条的显示信息
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_makespan = np.mean(recent_makespans) if recent_makespans else 0.0

        episode_pbar.set_postfix({
            "Avg Reward (100)": f"{avg_reward:.2f}",
            "Avg Makespan (100)": f"{avg_makespan:.2f}",
            "LR": f"{scheduler.get_last_lr()[0]:.1e}"
        })

        # --- 写入TensorBoard日志 ---
        writer.add_scalar("charts/makespan", final_makespan, episode)
        writer.add_scalar("charts/avg_fidelity", final_avg_fidelity, episode)
        writer.add_scalar("charts/total_crosstalk", final_total_crosstalk, episode)
        writer.add_scalar("hyperparameters/learning_rate", scheduler.get_last_lr()[0], episode)

        writer.add_scalar("physics/avg_swaps_per_task", avg_swaps, episode)
        writer.add_scalar("physics/avg_task_fidelity", avg_final_fidelity, episode)
        writer.add_scalar("physics/avg_crosstalk_per_task", avg_crosstalk, episode)

        # 每隔N个episode，或者在训练结束时，画一张图
        if episode % 200 == 0 or episode == args.MAX_EPISODES - 1:
            # a. 保存调度图
            if env.schedule_plan:
                plot_save_path = os.path.join(plots_dir, f"schedule_episode_{episode}.png")
                plot_schedule(env.schedule_plan, args.CHIP_SIZE, baseline_makespan, save_path=plot_save_path)

            # b. 保存模型权重
            checkpoint_path = os.path.join(checkpoints_dir, f"model_episode_{episode}.pth")
            torch.save(model.state_dict(), checkpoint_path)

            # 在tqdm进度条上打印一条清晰的保存信息
            episode_pbar.write(f"--- Checkpoint saved to {checkpoint_path} at episode {episode} ---")

        # 4. 在每个Episode结束时更新学习率
        scheduler.step()

        # =====================================================================
        # === 验证阶段 (Validation Phase) ===
        # =====================================================================

        # 每隔200个episodes (且不是第一个episode)，进行一次验证
        if episode > 0 and episode % 200 == 0:
            print(f"\n--- [Episode {episode}] Running Validation ---")

            # 将模型设置为评估模式
            model.eval()

            validation_makespans = []
            # 运行N次独立的验证episodes，以获得更稳定的平均性能
            num_validation_episodes = 20
            # 运行20次独立的验证episodes
            for _ in tqdm(range(num_validation_episodes), desc="Validating", leave=False):
                # 告诉环境从'validation'池中采样
                val_obs_dict, _ = env.reset(options={'pool': 'validation'})

                # 在验证时，我们不计算梯度
                with torch.no_grad():
                    # --- 运行一个完整的评估Episode ---
                    for _ in range(env.num_tasks):
                        # === 1. 任务选择 (GNN Selector - 贪心模式) ===
                        candidate_tasks = env.get_candidate_tasks()
                        if not candidate_tasks: break

                        chip_graph_data = env.get_chip_state_graph_data()

                        # 如果GNN选择器存在，则使用它
                        if gnn_selector:
                            task_graph_data_list = [networkx_to_pyg_data(t.interaction_graph) for t in candidate_tasks]

                            chip_batch = Batch.from_data_list([chip_graph_data] * len(candidate_tasks)).to(device)
                            task_batch = Batch.from_data_list(task_graph_data_list).to(device)

                            scores = gnn_selector(chip_batch, task_batch).cpu().numpy().flatten()

                            # 选择分数最高的任务
                            chosen_task_local_index = np.argmax(scores)
                            chosen_task = candidate_tasks[chosen_task_local_index]
                        else:
                            # 降级为随机选择
                            chosen_task = np.random.choice(candidate_tasks)

                        val_task_id = chosen_task.id

                        # === 2. 任务放置 (RL Placer - 贪心模式) ===
                        val_task_id = chosen_task.id
                        val_current_task = env.task_pool[val_task_id]
                        val_placement_in_progress = {}

                        # --- 微观放置循环 ---
                        for i in range(val_current_task.num_qubits):
                            # a. 获取观察
                            obs_dict_step = env._get_obs(val_task_id, val_placement_in_progress)
                            obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in
                                          obs_dict_step.items()}

                            # b. 模型决策
                            logits, _ = model(obs_tensor)

                            # c. 应用与训练时完全相同的硬约束掩码
                            # c.1 获取已占用的比特掩码
                            placement_mask = torch.tensor(obs_dict_step["placement_mask"], dtype=torch.bool).to(device)

                            # c.2 获取连通性约束掩码
                            connectivity_mask = torch.ones_like(placement_mask)
                            current_logical_idx = i
                            task_graph = val_current_task.interaction_graph

                            if i == 0:
                                # 第一个比特可以放在任何未占用的地方
                                connectivity_mask = placement_mask.clone()
                            else:
                                valid_neighbors = set()
                                # 检查当前逻辑比特 l_i 是否需要与任何已放置的比特 l_j 连接
                                needs_connection = any(task_graph.has_edge(current_logical_idx, j) for j in
                                                       val_placement_in_progress.keys())

                                if not needs_connection:
                                    # 如果 l_i 与已放置的部分没有连接，可以放在任何未占用的地方
                                    connectivity_mask = placement_mask.clone()
                                else:
                                    # 否则，它必须放在已放置邻居的空闲邻居位置
                                    for placed_logical_idx, placed_physical_id in val_placement_in_progress.items():
                                        if task_graph.has_edge(current_logical_idx, placed_logical_idx):
                                            for neighbor_id in env.chip_model[placed_physical_id].connectivity:
                                                neighbor_idx = env.qubit_id_to_idx[neighbor_id]
                                                if not placement_mask[neighbor_idx]:
                                                    valid_neighbors.add(neighbor_idx)

                                    if valid_neighbors:
                                        indices = torch.tensor(list(valid_neighbors), dtype=torch.long, device=device)
                                        connectivity_mask.scatter_(0, indices, 0)  # 将合法位置的掩码设为False
                                    else:
                                        # 如果所有邻居都被占用了，允许放在任何未占用的地方，依赖奖励惩罚
                                        connectivity_mask = placement_mask.clone()

                            # c.3 合并掩码
                            final_mask = placement_mask | connectivity_mask
                            logits[0, final_mask] = -1e8

                            # d. 【核心区别】使用 argmax 进行贪心选择
                            action = torch.argmax(logits, dim=1)

                            # e. 更新本次评估中的放置方案
                            physical_qubit_id = env.idx_to_qubit_id[action.item()]
                            val_placement_in_progress[i] = physical_qubit_id

                        # === 3. 提交计划 ===
                        final_mapping = val_placement_in_progress
                        start_time = max(env.chip_model[pid].get_next_available_time() for pid in
                                         final_mapping.values()) if final_mapping else 0.0

                        val_full_action = {"task_id": val_task_id, "mapping": final_mapping, "start_time": start_time}

                        _, _, val_terminated, _, _ = env.step(val_full_action)

                        if val_terminated:
                            break

                # 记录这次验证的makespan
                if env.schedule_plan:
                    validation_makespans.append(max(t['end_time'] for t in env.schedule_plan))

            # 计算平均验证makespan
            avg_validation_makespan = np.mean(validation_makespans) if validation_makespans else float('inf')
            writer.add_scalar("validation/avg_makespan", avg_validation_makespan, episode)
            print(
                f"--- Avg Validation Makespan: {avg_validation_makespan:.2f} (Best: {best_validation_makespan:.2f}) ---")

            # 只保存表现最好的模型
            if avg_validation_makespan < best_validation_makespan:
                best_validation_makespan = avg_validation_makespan
                best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"--- New best model saved to {best_model_path} ---")

            # 将模型切换回训练模式
            model.train()

    episode_pbar.close()
    # 关闭writer
    writer.close()
    print("Training finished.")


def update_ppo(model, optimizer, buffer, args, device, last_value, writer, global_step):
    """
    一个PPO更新函数，增加了TensorBoard日志记录。
    """
    # --- 计算优势 (GAE) ---  A_t 告诉我们在状态s_t下，采取动作a_t比平均水平好多少。V_target_t是Critic网络应该学习的目标。
    advantages = np.zeros(len(buffer["rewards"]), dtype=np.float32)
    last_gae_lam = 0
    last_value_np = last_value.cpu().detach().numpy().flatten()

    for t in reversed(range(len(buffer["rewards"]))):
        if t == len(buffer["rewards"]) - 1:
            next_non_terminal = 1.0 - buffer["dones"][t]
            next_values = last_value_np
        else:
            next_non_terminal = 1.0 - buffer["dones"][t + 1]
            next_values = buffer["values"][t + 1]

        delta = buffer["rewards"][t] + args.GAMMA * next_values * next_non_terminal - buffer["values"][t]
        advantages[t] = last_gae_lam = delta + args.GAMMA * args.GAE_LAMBDA * next_non_terminal * last_gae_lam
    returns = advantages + np.array(buffer["values"]).flatten()

    # --- 转换为Tensor ---
    obs_tensors = {k: torch.tensor(np.array([d[k] for d in buffer["obs"]]), dtype=torch.float32).to(device) for k in buffer["obs"][0]}
    actions_t = torch.tensor(buffer["actions"]).flatten().to(device)
    log_probs_old_t = torch.tensor(buffer["log_probs"]).flatten().to(device)
    advantages_t = torch.tensor(advantages).flatten().to(device)
    returns_t = torch.tensor(returns).flatten().to(device)

    # --- 多轮次小批量更新 ---
    indices = np.arange(len(buffer["rewards"]))
    for i in range(args.PPO_EPOCHS): # i 是 PPO epoch 的索引
        np.random.shuffle(indices)
        for start in range(0, len(buffer["rewards"]), args.MINI_BATCH_SIZE):
            end = start + args.MINI_BATCH_SIZE
            batch_indices = indices[start:end]

            # ... (提取小批量数据) ...
            batch_obs = {k: v[batch_indices] for k, v in obs_tensors.items()}
            batch_actions = actions_t[batch_indices]
            # 收集数据时，旧策略对所选动作的打分
            batch_log_probs_old = log_probs_old_t[batch_indices]
            batch_advantages = advantages_t[batch_indices]
            # 我们在第一部分计算出的 “正确答案” （实际价值目标）。
            batch_returns = returns_t[batch_indices]

            # new_values: Critic网络对于小批量数据中的状态，做出的新预测。
            logits, new_values = model(batch_obs)
            new_probs = Categorical(logits=logits)
            # 更新网络时，新策略对同一个动作的打分
            new_log_probs = new_probs.log_prob(batch_actions)
            entropy = new_probs.entropy()

            # --- 计算Loss ---
            # 概率比率 衡量了新策略相对于旧策略，有多么“倾向于”做出我们当时实际做出的那个动作 a，ratio > 1: 新策略更喜欢这个动作
            ratio = torch.exp(new_log_probs - batch_log_probs_old)
            # 传统的策略梯度目标
            # 如果advantage > 0（好动作），我们就想提高ratio（即提高选择这个动作的概率），从而最大化这个目标。
            surr1 = ratio * batch_advantages
            # PPO的“安全带”,clamp(ratio, 1 - ε, 1 + ε): 我们将ratio强制限制在一个很小的区间内，例如[0.8, 1.2]
            # 如果advantage > 0（好动作），我们鼓励ratio变大，但最大只能到1.2。我们不允许新策略因为一个好动作就变得过于“激动”，更新幅度被限制了。
            surr2 = torch.clamp(ratio, 1 - args.PPO_EPSILON, 1 + args.PPO_EPSILON) * batch_advantages
            # 取这两个目标中更悲观（更小）的那个
            # 当advantage > 0时，surr2限制了更新的上限。
            # 当advantage < 0时，surr1可能比surr2更小，这意味着如果ratio变得过大（新策略突然极度不喜欢一个坏动作），我们允许一次幅度更大的更新。
            # 最后的负号，是因为优化器是最小化loss，而我们的目标是最大化策略表现。
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)
            entropy_value = entropy.mean()
            loss = actor_loss + args.CRITIC_DISCOUNT * critic_loss - args.ENTROPY_BETA * entropy_value

            # --- 更新 ---
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    # --- 在所有PPO Epochs结束后记录一次最终的Loss值 ---
    # 这样可以避免图表过于密集，我们只关心每个Rollout更新后的最终状态
    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
    writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_value.item(), global_step)
    writer.add_scalar("losses/total_loss", loss.item(), global_step)


if __name__ == "__main__":
    main()
