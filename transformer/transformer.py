""" Transformer模型和编码器，解码器 """

import torch.nn as nn

from transformer.layers import EncoderLayer, DecoderLayer
from transformer.pos_encoder import PositionEncoder


class Transformer(nn.Module):
    def __init__(self, layer_num, head_num, word_vec, d_ff, src_vocab_size, trg_vocab_size, max_seq_size, src_pad_idx, trg_pad_idx, dropout):
        """
        初始化Transformer模型，包括位置编码，词嵌入模型等

        :param layer_num: EncoderLayer和DecoderLayer的层数
        :param head_num: 多头注意力中的头数
        :param word_vec: 词向量维度大小
        :param d_ff: 全连接层的映射维度
        :param src_vocab_size: 源序列词表数
        :param trg_vocab_size: 目标序列词表数
        :param max_seq_size: 最长序列数
        :param src_pad_idx: 源序列填充词ID
        :param trg_pad_idx: 目标序列填充词ID
        :param dropout: 随机失活率
        """
        super().__init__()
        self.position_encoder = PositionEncoder(max_seq_size, word_vec)
        self.encoder_embedding = nn.Embedding(src_vocab_size, word_vec, padding_idx=src_pad_idx)
        self.encoder = nn.ModuleList([EncoderLayer(head_num, word_vec, d_ff, dropout) for _ in range(layer_num)])
        self.decoder_embedding = nn.Embedding(trg_vocab_size, word_vec, padding_idx=trg_pad_idx)
        self.decoder = nn.ModuleList([DecoderLayer(head_num, word_vec, d_ff, dropout) for _ in range(layer_num)])
        self.output = nn.Sequential(
            nn.Linear(word_vec, trg_vocab_size),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src_seq, trg_seq):
        """
        Transformer模型的前向计算

        :param src_seq: 源序列，经过token之后的单词索引
        :param trg_seq: 目标序列，经过token之后的单词索引
        :return: Transformer模型输出，表示第i个时间点预测词表j的概率
        """
        # ==================== 初始化src和trg的padding mask ====================
        src_pad_mask = src_seq.eq(self.src_pad_idx)
        trg_pad_mask = trg_seq.eq(self.trg_pad_idx)

        # ==================== Embedding和位置编码 ====================
        # 计算位置编码
        src_seq_input = self.encoder_embedding(src_seq) + self.position_encoder.position_codes
        trg_seq_input = self.decoder_embedding(trg_seq) + self.position_encoder.position_codes

        # 添加必要的dropout
        src_seq_input = self.dropout(src_seq_input)
        trg_seq_input = self.dropout(trg_seq_input)

        # todo: 考虑是否需要在dropout外面再加一层 layer_norm

        # ==================== 编码器处理 ====================
        for encoder_layer in self.encoder:
            src_seq_input = encoder_layer(src_seq_input, src_pad_mask)

        # ==================== 解码器处理 ====================
        for decoder_layer in self.decoder:
            trg_seq_input = decoder_layer(src_seq_input, trg_seq_input, src_pad_mask, trg_pad_mask)

        # ==================== 输出层处理 ====================
        output = self.output(trg_seq_input)
        return output
