import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# --- 1. 导入自定义模块 ---
from environment import QuantumSchedulingEnv
from models.quantum_task import QuantumTask  # 确保__init__.py工作正常
from torch.utils.tensorboard import SummaryWriter # 导入TensorBoard的Writer
import time
import os # 导入os模块
from visualizer import plot_schedule # 导入我们的绘图函数

# --- 2. 定义超参数 ---
class Hyperparameters:
    # 环境参数
    CHIP_SIZE = (5, 5)
    NUM_TASKS = 10
    MAX_QUBITS_PER_TASK = 6
    # PPO 训练参数
    LEARNING_RATE = 5e-4
    ENTROPY_BETA = 0.02
    GAMMA = 0.99  # 折扣因子
    GAE_LAMBDA = 0.95  # GAE平滑参数
    PPO_EPSILON = 0.2  # PPO裁剪系数
    CRITIC_DISCOUNT = 0.5  # Critic loss的系数
    PPO_EPOCHS = 4  # 每次更新时，用同一批数据训练的次数
    MINI_BATCH_SIZE = 64
    ROLLOUT_LENGTH = 2048  # 收集多少步经验后进行一次网络更新
    # 训练过程参数
    MAX_EPISODES = 1000


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

    env = QuantumSchedulingEnv(
        chip_size=args.CHIP_SIZE,
        num_tasks=args.NUM_TASKS,
        max_qubits_per_task=args.MAX_QUBITS_PER_TASK
    )

    action_space_dim = env.num_qubits  # 我们的动作是选择一个物理比特
    model = ActorCritic(env.observation_space, action_space_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

    # PPO经验缓冲区
    rollout_buffer = {
        "obs": [], "actions": [], "log_probs": [], "rewards": [],
        "dones": [], "values": []
    }

    # --- 训练循环 ---
    global_step = 0
    # 在main函数开始时，计算一个基线
    baseline_makespan = sum(t.estimated_duration for t in env.task_pool.values())
    for episode in range(args.MAX_EPISODES):
        # --- 宏观循环: 调度一个完整的方案 ---
        # 在Episode开始前，记录一下最终的统计数据
        final_makespan = 0
        final_avg_fidelity = 0
        final_total_crosstalk = 0

        initial_obs_dict, info = env.reset()
        episode_reward = 0

        for task_id in range(env.num_tasks):
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

                # 应用掩码，防止选择已占用的比特
                placement_mask = torch.tensor(obs_dict["placement_mask"], dtype=torch.bool).to(device)
                logits[0, placement_mask] = -1e8  # 将非法动作的logit设为极小值

                probs = Categorical(logits=logits) # PyTorch 的 Categorical 分布对象，把 logits 转换成概率分布
                action = probs.sample()  # 采样一个动作 (物理比特的索引)
                log_prob = probs.log_prob(action)

                # 3. 存储经验
                rollout_buffer["obs"].append(obs_dict)
                rollout_buffer["actions"].append(action.cpu().numpy())
                rollout_buffer["log_probs"].append(log_prob.cpu().numpy())
                rollout_buffer["values"].append(value.cpu().numpy())
                rollout_buffer["rewards"].append(0)  # 中间奖励为0
                rollout_buffer["dones"].append(False)

                # 4. 更新进行中的放置
                physical_qubit_id = env.idx_to_qubit_id[action.item()]  # 将action的下表（例如2）转换为qubit_id 例如(2,3)
                placement_in_progress[i] = physical_qubit_id

                # 5. 检查是否需要更新网络
                if len(rollout_buffer["actions"]) >= args.ROLLOUT_LENGTH:
                    update_ppo(model, optimizer, rollout_buffer, args, device, last_value=value)
                    # 清空缓冲区
                    for k in rollout_buffer: rollout_buffer[k] = []

            # --- 放置完成后，提交计划给环境 ---
            final_mapping = placement_in_progress
            start_time = 0
            if final_mapping:
                start_time = max(env.chip_model[pid].get_next_available_time() for pid in final_mapping.values())

            full_action = {"task_id": task_id, "mapping": final_mapping, "start_time": start_time}
            _, reward, terminated, _, _ = env.step(full_action)
            episode_reward += reward  # reward只在最后一步非0

        # --- Episode 结束 ---
        # 更新最后一步的奖励和完成状态
        if rollout_buffer["rewards"]:
            rollout_buffer["rewards"][-1] = episode_reward
            rollout_buffer["dones"][-1] = True

        # --- 计算并记录最终的调度方案指标 ---
        if env.schedule_plan:
            final_makespan = max(t['end_time'] for t in env.schedule_plan)
            # fidelities = [t['fidelity'] for t in env.schedule_plan]
            # final_avg_fidelity = np.prod(fidelities) ** (1 / len(fidelities)) if fidelities else 0
            # final_total_crosstalk = sum(t['crosstalk'] for t in env.schedule_plan)

        print(f"Episode {episode}: Total Reward = {episode_reward:.4f}")

        # 每隔N个episode，或者在训练结束时，画一张图
        if episode % 50 == 0 or episode == args.MAX_EPISODES - 1:
            if env.schedule_plan:
                # 定义图片的保存路径
                plot_save_path = os.path.join(plots_dir, f"schedule_episode_{episode}.png")
                # 调用绘图函数并传入路径
                plot_schedule(env.schedule_plan, args.CHIP_SIZE, baseline_makespan,plot_save_path)
        # --- 写入TensorBoard日志 ---
        writer.add_scalar("charts/episode_reward", episode_reward, episode)
        writer.add_scalar("charts/makespan", final_makespan, episode)
        writer.add_scalar("charts/avg_fidelity", final_avg_fidelity, episode)
        writer.add_scalar("charts/total_crosstalk", final_total_crosstalk, episode)
        writer.add_scalar("charts/learning_rate", args.LEARNING_RATE, episode)  # 记录学习率
        # 在update_ppo函数中，我们也可以记录loss
        # writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        # ...
        env.render()

    # 关闭writer
    writer.close()
    print("Training finished.")

def update_ppo(model, optimizer, buffer, args, device, last_value):
    """一个简化的PPO更新函数"""
    # --- 计算优势 (GAE) ---
    advantages = np.zeros(len(buffer["rewards"]), dtype=np.float32)
    last_gae_lam = 0

    # 将last_value转换为numpy
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
    obs_tensors = {k: torch.tensor(np.array([d[k] for d in buffer["obs"]]), dtype=torch.float32).to(device) for k in
                   buffer["obs"][0]}
    actions_t = torch.tensor(buffer["actions"]).flatten().to(device)
    log_probs_old_t = torch.tensor(buffer["log_probs"]).flatten().to(device)
    advantages_t = torch.tensor(advantages).flatten().to(device)
    returns_t = torch.tensor(returns).flatten().to(device)

    # --- 多轮次小批量更新 ---
    indices = np.arange(len(buffer["rewards"]))
    for _ in range(args.PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, len(buffer["rewards"]), args.MINI_BATCH_SIZE):
            end = start + args.MINI_BATCH_SIZE
            batch_indices = indices[start:end]

            # 提取小批量数据
            batch_obs = {k: v[batch_indices] for k, v in obs_tensors.items()}
            batch_actions = actions_t[batch_indices]
            batch_log_probs_old = log_probs_old_t[batch_indices]
            batch_advantages = advantages_t[batch_indices]
            batch_returns = returns_t[batch_indices]

            # 重新计算新策略下的值
            logits, new_values = model(batch_obs)
            new_probs = Categorical(logits=logits)
            new_log_probs = new_probs.log_prob(batch_actions)
            entropy = new_probs.entropy()

            # 计算Actor Loss
            ratio = torch.exp(new_log_probs - batch_log_probs_old)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - args.PPO_EPSILON, 1 + args.PPO_EPSILON) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算Critic Loss
            critic_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)

            # 计算总Loss
            loss = actor_loss + args.CRITIC_DISCOUNT * critic_loss - args.ENTROPY_BETA * entropy.mean()

            # 更新
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
            optimizer.step()


if __name__ == "__main__":
    main()
