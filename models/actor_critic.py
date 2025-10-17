# 文件: models/actor_critic.py
from typing import Any

import torch
import torch.nn as nn
import numpy as np


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
        qubit_input_dim = obs_space["qubit_embeddings"].shape[-1]
        task_input_dim = obs_space["task_embeddings"].shape[-1]
        context_input_dim = obs_space["logical_qubit_context"].shape[-1]

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
            batch_first=True  # 确保输入形状是 (Batch, SeqLen, Features)
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

        # 任务选择头 (Task Selection Head)
        # 它只依赖于全局特征和所有任务的语境化特征
        self.task_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Actor放置头：根据全局特征和比特特征共同决策
        # 输入维度 = 全局特征(128) + 芯片总特征(d_model * num_qubits)
        num_qubits = obs_space["qubit_embeddings"].shape[0]
        placement_input_dim = 128 + (d_model * num_qubits)
        self.placement_head = nn.Sequential(
            nn.Linear(placement_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_dim)
        )

        # Critic头：只依赖于全局特征
        self.critic_head = nn.Linear(128, 1)

    def forward(self, obs: dict) -> tuple[Any, Any, Any]:
        batch_size = obs["qubit_embeddings"].shape[0]

        # a. 将所有不同来源、不同维度的特征，通过线性层（nn.Linear）统一映射到 Transformer 能理解的维度 d_model (128)。
        qubit_embeds = self.qubit_embed(obs["qubit_embeddings"])  # (B, NumQubits, D_model)
        task_embed = self.task_embed(obs["task_embeddings"])
        context_embed = self.context_embed(obs["logical_qubit_context"]).unsqueeze(1)  # (B, 1, D_model)

        # b. 构建完整的输入序列
        # [CLS], [TASK], [CONTEXT], [QUBIT_1], ..., [QUBIT_25]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # shape: (B, 1, 128)
        # 拼接
        full_sequence = torch.cat([
            cls_tokens,
            task_embed,
            context_embed,
            qubit_embeds
        ], dim=1)

        # c. 创建Padding Mask
        task_mask = obs["task_mask"].bool()
        special_tokens_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=task_mask.device)
        qubit_mask = torch.zeros(batch_size, qubit_embeds.shape[1], dtype=torch.bool, device=task_mask.device)
        src_key_padding_mask = torch.cat([
            special_tokens_mask, task_mask, qubit_mask
        ], dim=1)

        # d. 通过Encoder
        encoded_sequence = self.transformer_encoder(full_sequence, src_key_padding_mask=src_key_padding_mask)

        # e. 提取特征
        global_features = encoded_sequence[:, 0]
        mlp_output = self.shared_mlp(global_features)

        # --- Critic 输出 ---
        state_value = self.critic_head(mlp_output)

        # --- Actor 输出 ---
        # 1. 计算任务选择 Logits
        num_tasks = obs["task_embeddings"].shape[1]
        # 序列构成: [CLS], [CONTEXT], [TASKS...], [QUBITS...]
        task_features = encoded_sequence[:, 2: 2 + num_tasks]
        task_logits = self.task_head(task_features).squeeze(-1)

        # 2. 计算放置 Logits
        qubit_features = encoded_sequence[:, 2 + num_tasks:]
        qubit_features_flat = qubit_features.flatten(start_dim=1)

        actor_placement_input = torch.cat([mlp_output, qubit_features_flat], dim=1)
        # 现在 self.placement_head 已经被正确定义
        placement_logits = self.placement_head(actor_placement_input)

        return task_logits, placement_logits, state_value
