""" 位置编码 """
import torch


class PositionEncoder:
    def __init__(self, seq_size, word_vec):
        """
        根据正弦余弦函数生成位置编码，编码和输入向量无关，只和单词位置pos和词向量维度dim有关，并在类初始化时预计算位置编码，避免循环调用计算同样逻辑

        :param seq_size: 句子序列长度
        :param word_vec: 词向量维度
        """
        self.seq_size = seq_size
        self.word_vec = word_vec
        self.position_codes = self.generate_position_codes()


    def generate_position_codes(self):
        """
        生成位置编码矩阵，采用torch.meshgrid避免张量矩阵大量循环遍历，提升效率

        :return: 位置编码矩阵
        """
        position_codes = torch.empty(self.seq_size, self.word_vec)
        pos_grid, dim_grad = torch.meshgrid(
            torch.arange(position_codes.size(0), dtype=torch.float),
            torch.arange(position_codes.size(1), dtype=torch.float),
            indexing='ij'
        )
        position_codes = self.generate_position_code(pos_grid, dim_grad)
        return position_codes

    def generate_position_code(self, pos_grid, dim_grad):
        """
        生成矩阵中某个元素的位置编码，使用torch进行向量化操作

        :param pos_grid: 位置行网格，表示词所在位置
        :param dim_grad: 位置列网格，表示词向量维度
        :return: 某个元素的位置编码
        """
        dim_value = torch.floor(dim_grad / 2)
        even_mask = (dim_grad % 2 == 0)
        coding = torch.where(
            even_mask,
            torch.sin(pos_grid / torch.pow(10000, 2.0 * dim_value / self.word_vec)),
            torch.cos(pos_grid / torch.pow(10000, 2.0 * dim_value / self.word_vec))
        )
        return coding
