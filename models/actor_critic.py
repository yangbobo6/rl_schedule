# 文件: models/actor_critic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class CNNActorCritic(nn.Module):
    """
    一个Actor-Critic模型，使用CNN处理芯片状态，并结合任务特征进行决策。
    """

    def __init__(self, obs_space, action_space_dim):
        super(CNNActorCritic, self).__init__()

        # 获取各部分观察的维度
        qubit_channels = obs_space["qubit_embeddings"].shape[1]
        chip_dim = int(np.sqrt(obs_space["qubit_embeddings"].shape[0]))  # 5x5 -> 5
        task_embed_dim = obs_space["current_task_embedding"].shape[0]
        print(f"Model Init: task_embed_dim={task_embed_dim}")
        placement_mask_dim = obs_space["placement_mask"].shape[0]
        logic_context_dim = obs_space["logical_qubit_context"].shape[0]

        # 1. CNN 用于处理芯片状态 (grid-like data)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=qubit_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算CNN输出的维度
        cnn_output_dim = self._get_cnn_output_dim(chip_dim, self.cnn)

        # 2. 拼接CNN输出和其他特征后的MLP
        combined_features_dim = cnn_output_dim + task_embed_dim + placement_mask_dim + logic_context_dim

        self.shared_mlp = nn.Sequential(
            nn.Linear(combined_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 3. Actor 和 Critic 头
        self.actor_head = nn.Linear(128, action_space_dim)
        self.critic_head = nn.Linear(128, 1)

    def _get_cnn_output_dim(self, chip_dim: int, cnn: nn.Module) -> int:
        """辅助函数，用于动态计算CNN展平后的输出维度"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, cnn[0].in_channels, chip_dim, chip_dim)
            output = cnn(dummy_input)
        return output.shape[1]

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # a. 处理芯片状态 (Reshape and CNN)
        qubit_embeds = obs["qubit_embeddings"]
        # (Batch, NumQubits, Channels) -> (Batch, Channels, Height, Width)
        chip_dim = int(np.sqrt(qubit_embeds.shape[1]))
        chip_image = qubit_embeds.view(qubit_embeds.shape[0], chip_dim, chip_dim, -1).permute(0, 3, 1, 2)
        chip_features = self.cnn(chip_image)

        # b. 拼接所有特征
        combined = torch.cat([
            chip_features,
            obs["current_task_embedding"],
            obs["placement_mask"].float(),
            obs["logical_qubit_context"]
        ], dim=1)

        # c. 通过共享MLP和输出头
        shared_output = self.shared_mlp(combined)
        action_logits = self.actor_head(shared_output)
        state_value = self.critic_head(shared_output)

        return action_logits, state_value