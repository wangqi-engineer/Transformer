""" 多头注意力模块实现 """

import torch
import torch.nn as nn

from transformer.attention import Attention
from transformer.model_constants import CROSS_ATTENTION_TYPE

__author__ = "Wang Qi"

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, word_vec, attention_type, dropout):
        """
        初始化多头注意力类

        :param head_num: 多头注意力中的头数
        :param word_vec: 词向量维度
        :param attention_type: 注意力类型
        :param dropout: 随机失活率
        """
        super().__init__()
        self.head_num = head_num
        self.word_vec = word_vec
        self.w_o = nn.Linear(word_vec, word_vec)
        self.dim_size = int(word_vec / head_num)
        self.attention_type = attention_type
        self.self_attentions = nn.ModuleList([Attention(word_vec, self.dim_size, dropout=dropout, attention_type=attention_type) for _ in range(head_num)])

    def forward(self, x, pad_mask, output_dec=None):
        """
        多头注意力的计算机制，主要是对每个头的注意力进行拼接和线性合并

        :param x: 输入，当为交叉注意力时表示输入来自编码器
        :param pad_mask: 标记src序列位置是否是pad词，如果是pad词在计算attention时不需要关注
        :param output_dec: 自注意力和掩码注意力时output_dec默认为None, 交叉注意力时表示输入来自解码器
        :return: 多头注意力输出
        """
        # 通过预定义tensor的大小并循环赋值的方式实现对多头注意力的concat
        concat_output = torch.zeros_like(x)
        for idx, self_attention in enumerate(self.self_attentions):
            if self.attention_type != CROSS_ATTENTION_TYPE:
                head_i_output = self_attention(x, pad_mask=pad_mask)
            else:
                head_i_output = self_attention(x, input_dec=output_dec, pad_mask=pad_mask)
            head_dim = head_i_output.size(-1)
            start_idx = idx * head_dim
            end_idx = (idx + 1) * head_dim
            # 避免在循环中使用torch.concat或者cat，效率太低且消耗大量内存
            # 这样拼接的前提是每个头的维度是一样的
            concat_output[:, :, start_idx:end_idx] = head_i_output
        output = self.w_o(concat_output)
        return output
