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


class TransformerActorCritic(nn.Module):
    """
    一个基于Transformer Encoder的Actor-Critic模型。
    它能处理结构化的输入，并学习特征之间的全局依赖关系。
    """

    def __init__(self, obs_space, action_space_dim, d_model=128, nhead=4, num_encoder_layers=3):
        super(TransformerActorCritic, self).__init__()

        self.d_model = d_model

        # --- 1. 输入嵌入层 ---
        # 将不同来源的特征统一映射到模型的隐藏维度 d_model
        qubit_input_dim = obs_space["qubit_embeddings"].shape[1]
        task_input_dim = obs_space["current_task_embedding"].shape[0]
        context_input_dim = obs_space["logical_qubit_context"].shape[0]

        self.qubit_embed = nn.Linear(qubit_input_dim, d_model)
        self.task_embed = nn.Linear(task_input_dim, d_model)
        self.context_embed = nn.Linear(context_input_dim, d_model)

        # placement_mask 不需要嵌入，它将在注意力模块中作为掩码使用

        # --- 2. Transformer Encoder 主干 ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,  # 通常是d_model的2-4倍
            dropout=0.1,
            batch_first=True  # 非常重要，确保输入形状是 (Batch, SeqLen, Features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- 3. 特殊的 [CLS] Token ---
        # 这是一个可学习的向量，用于聚合整个序列的全局信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # --- 4. 输出头 ---
        # 共享的MLP，处理来自 [CLS] token的全局特征
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(d_model),  # 在输入MLP前进行归一化
            nn.Linear(d_model, 128),
            nn.ReLU()
        )

        # Actor头：根据全局特征和比特特征共同决策
        # 输入维度 = 全局特征(128) + 芯片总特征(d_model * num_qubits)
        num_qubits = obs_space["qubit_embeddings"].shape[0]
        actor_input_dim = 128 + d_model * num_qubits
        self.actor_head = nn.Sequential(
            nn.Linear(actor_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_dim)
        )

        # Critic头：只依赖于全局特征
        self.critic_head = nn.Linear(128, 1)

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs["qubit_embeddings"].shape[0]

        # a. 将各个输入部分进行嵌入
        qubit_embeds = self.qubit_embed(obs["qubit_embeddings"])  # (B, NumQubits, D_model)
        task_embed = self.task_embed(obs["current_task_embedding"]).unsqueeze(1)  # (B, 1, D_model)
        context_embed = self.context_embed(obs["logical_qubit_context"]).unsqueeze(1)  # (B, 1, D_model)

        # b. 构建完整的输入序列
        # [CLS], [TASK], [CONTEXT], [QUBIT_1], ..., [QUBIT_25]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        full_sequence = torch.cat([
            cls_tokens,
            task_embed,
            context_embed,
            qubit_embeds
        ], dim=1)

        # c. 通过Transformer Encoder
        # 注意：我们需要一个padding mask来告诉Transformer忽略哪些部分，但在这个固定长度输入的场景下可以简化
        encoded_sequence = self.transformer_encoder(full_sequence)

        # d. 提取特征用于决策
        # 全局特征来自 [CLS] token 的输出
        global_features = encoded_sequence[:, 0]  # (B, D_model)
        mlp_output = self.shared_mlp(global_features)  # (B, 128)

        # 芯片特征来自所有比特的输出
        qubit_features = encoded_sequence[:, 3:].flatten(start_dim=1)  # (B, NumQubits * D_model)

        # e. 通过输出头
        # Actor的决策依赖于全局情况和芯片的具体情况
        actor_input = torch.cat([mlp_output, qubit_features], dim=1)
        action_logits = self.actor_head(actor_input)

        # Critic的评估只依赖于全局情况
        state_value = self.critic_head(mlp_output)

        return action_logits, state_value
