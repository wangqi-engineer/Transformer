""" 编码器和解码器子层 """
import torch.nn as nn

from transformer.model_constants import SELF_ATTENTION_TYPE, MASKED_ATTENTION_TYPE, CROSS_ATTENTION_TYPE
from transformer.layer_norm import LayerNorm
from transformer.heads_attention import MultiHeadAttention

__author__ = "Wang Qi"

class EncoderLayer(nn.Module):
    """ 编码器子层 """

    def __init__(self, head_num, word_vec, d_ff, dropout):
        """
        初始化EncoderLayer类

        :param head_num: 多头注意力中的头数
        :param word_vec: 词向量大小
        :param d_ff: 全连接层的映射维度
        :param dropout: 随机失活率
        """
        super().__init__()
        self.self_attentions = MultiHeadAttention(head_num, word_vec, dropout=dropout, attention_type=SELF_ATTENTION_TYPE)
        self.feed_forward = nn.Sequential(
            nn.Linear(word_vec, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, word_vec)
        )
        self.norms = nn.ModuleList([LayerNorm(word_vec) for _ in range(2)])
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(2))


    def forward(self, x, pad_mask):
        """
        EncoderLayer类计算机制，采用Pre-LN思路

        :param x: 输入
        :param pad_mask: 标记当前序列位置是否是pad词，如果是pad词在计算attention时不需要关注
        :return: EncoderLayer层的输出
        """
        # ==================== 多头自注意力机制 ====================
        # pre-ln，在一开始就进行layer_norm
        layer_norm_x = self.norms[0](x)
        attentions_output = self.self_attentions(layer_norm_x, pad_mask=pad_mask)
        attentions_output = self.dropouts[0](attentions_output)

        # ==================== 残差连接 ====================
        residual_add = attentions_output + x

        # ==================== feed_forward层 ====================
        norm_output = self.norms[1](residual_add)
        feed_forward_output = self.feed_forward(norm_output)
        feed_forward_output = self.dropouts[1](feed_forward_output)

        # ==================== 残差连接 ====================
        output = feed_forward_output + residual_add
        return output


class DecoderLayer(nn.Module):
    """ 解码器子层 """

    def __init__(self, head_num, word_vec, d_ff, dropout):
        """
        初始化解码器子层类，采用Pre-LN思路

        :param head_num: 多头注意力机制的头数
        :param word_vec: 词向量大小
        :param d_ff: 全连接层的映射维度
        :param dropout: 随机失活率
        """
        super().__init__()
        self.masked_attentions = MultiHeadAttention(head_num, word_vec, dropout=dropout, attention_type=MASKED_ATTENTION_TYPE)
        self.cross_attentions = MultiHeadAttention(head_num, word_vec, dropout=dropout, attention_type=CROSS_ATTENTION_TYPE)
        self.feed_forward = nn.Sequential(
            nn.Linear(word_vec, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, word_vec)
        )
        self.norms = nn.ModuleList([LayerNorm(word_vec) for _ in range(3)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])

    def forward(self, input_enc, input_dec, src_pad_mask, trg_pad_mask):
        """
        DecoderLayer类计算机制

        :param input_enc: 输入来自编码器
        :param input_dec: 输入来自解码器
        :param src_pad_mask: 标记src序列位置是否是pad词，如果是pad词在计算attention时不需要关注
        :param trg_pad_mask: 标记trg序列位置是否是pad词，如果是pad词在计算attention时不需要关注
        :return:
        """
        # ==================== 多头掩码注意力机制 ====================
        # pre-ln，在一开始就进行layer_norm
        input_dec_norm = self.norms[0](input_dec)
        output_masked_attention = self.masked_attentions(input_dec_norm, pad_mask=trg_pad_mask)
        output_masked_attention = self.dropouts[0](output_masked_attention)

        # ==================== 残差连接 ====================
        masked_attention_res_add = output_masked_attention + input_dec

        # ==================== 多头交叉注意力机制 ====================
        masked_attention_norm_output = self.norms[1](masked_attention_res_add)
        output_cross_attention = self.cross_attentions(masked_attention_norm_output, output_dec=input_enc, pad_mask=src_pad_mask)
        output_cross_attention = self.dropouts[1](output_cross_attention)

        # ==================== 残差连接 ====================
        cross_attention_res_add = output_cross_attention + masked_attention_res_add

        # ==================== feed_forward层 ====================
        cross_attention_norm_output = self.norms[2](cross_attention_res_add)
        feed_forward_output = self.feed_forward(cross_attention_norm_output)
        feed_forward_output = self.dropouts[2](feed_forward_output)

        # ==================== 残差连接 ====================
        norm_output = cross_attention_res_add + feed_forward_output
        return norm_output
