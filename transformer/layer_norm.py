""" 归一化层 """

import torch
import torch.nn as nn

__author__ = "Wang Qi"

class LayerNorm(nn.Module):
    def __init__(self, word_vec, eps=1e-5):
        """
        初始化归一化层

        :param word_vec: 词向量大小，等价于输入维度的最后一个值
        :param eps: 最小量，避免归一化时除零异常，可选参数，给定默认值
        """
        super().__init__()
        self.model_size = word_vec
        self.eps = eps
        # 标准LayerNorm使用可学习的缩放和平移参数
        self.weight = nn.Parameter(torch.ones(word_vec))  # 缩放参数 γ
        self.bias = nn.Parameter(torch.zeros(word_vec))  # 平移参数 β

        # 专用初始化
        self._initialize_parameters()

    def _initialize_parameters(self):
        """专用参数初始化"""
        # LayerNorm的权重通常初始化为1，偏置初始化为0
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        归一化输入最后一维向量，并通过可学习参数缩放平移

        :param x: 输入
        :return: 归一化输出
        """
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        output = x_norm * self.weight + self.bias
        return output
