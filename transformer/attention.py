""" 注意力模块实现 """
import numpy as np
import torch
import torch.nn as nn

from transformer.model_constants import SELF_ATTENTION_TYPE, CROSS_ATTENTION_TYPE, MASKED_ATTENTION_TYPE


class Attention(nn.Module):
    def __init__(self, word_vec, model_size, dropout, attention_type=SELF_ATTENTION_TYPE):
        """
        初始化 Self-Attention 类

        :param word_vec: 词向量长度
        :param model_size: 模型大小
        :param attention_type: 注意力类型：包括自注意力，掩码注意力和交叉注意力
        """
        super().__init__()
        self.word_vec = word_vec
        self.model_size = model_size
        self.w_q = nn.Linear(word_vec, model_size)
        self.w_k = nn.Linear(word_vec, model_size)
        self.w_v = nn.Linear(word_vec, model_size)
        self.attention_type = attention_type

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask, input_dec=None):
        """
        attention注意力机制计算

        :param x: 输入，当attention_type为cross_attention时，表示源自编码器的输入
        :param pad_mask: 标记当前序列位置是否是pad词，如果是pad词在计算attention时不需要关注
        :param input_dec: 表示源自解码器的输入
        :return: self-attention 计算结果
        """
        # ==================== 生成Q, K, V矩阵 ====================
        # 如果是交叉注意力，需要修改Q矩阵的输入来源
        query_input = x if self.attention_type != CROSS_ATTENTION_TYPE else input_dec
        q = self.w_q(query_input)
        k = self.w_k(x)
        v = self.w_v(x)

        # ==================== 注意力分数矩阵 ====================
        s = q @ k.transpose(1, 2)

        # ==================== 点积缩放 ====================
        s_scaled = s / np.sqrt(self.model_size)

        # ==================== softmax归一化 ====================
        # 如果时pad词则加上一个很小的数，softmax会给该pad词分配的权重很小，避免噪声干扰
        s_scaled = s_scaled.masked_fill(pad_mask.unsqueeze(1), -1e9)

        # 如果是掩码注意力需要加上一个上三角矩阵
        if self.attention_type == MASKED_ATTENTION_TYPE:
            m = torch.triu(torch.ones_like(s_scaled), diagonal=1)
            s_scaled = s_scaled.masked_fill(m, -1e9)
        scores = torch.softmax(s_scaled, dim=-1)

        # 进行一次dropout
        scores = self.dropout(scores)

        # ==================== 计算输出矩阵 ====================
        output = scores @ v
        return output
