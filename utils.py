""" 将单词映射为id并转为tensor """
import torch
from tqdm import tqdm

from constants import UNK_WORD, BOS_WORD, EOS_WORD, PAD_WORD


class Tokenizer:
    def __init__(self, vocab):
        # 构建单词和索引的dict双向映射，通过dict加速查询速度
        self.vocab_s2i = {'src': {}, 'trg': {}}
        self.vocab_i2s = {'src': {}, 'trg': {}}

        for idx, word in enumerate(vocab['src']):
            self.vocab_s2i['src'][word] = idx
            self.vocab_i2s['src'][idx] = word

        for idx, word in enumerate(vocab['trg']):
            self.vocab_s2i['trg'][word] = idx
            self.vocab_i2s['trg'][idx] = word

    @staticmethod
    def _sentence_tokenize(sentence, vocab_s2i):
        words = sentence.split()
        # 从词表中直接查索引，查不到标记为<unk>
        unk_idx = vocab_s2i.get(UNK_WORD)
        tokens = [vocab_s2i.get(word, unk_idx) for word in words]
        return tokens

    def tokenize(self, data):
        src_list = []
        trg_list = []
        desc = '    - (Tokenizing) '
        # 先统一处理为dict再一次转tensor，提升性能
        for sentence in tqdm(data, desc=desc, mininterval=2, leave=True):
            src_sentence = self._sentence_tokenize(sentence['src'], self.vocab_s2i['src'])
            trg_sentence = self._sentence_tokenize(sentence['trg'], self.vocab_s2i['trg'])
            src_list.append(src_sentence)
            trg_list.append(trg_sentence)
        src_tensor = torch.tensor(src_list)
        trg_tensor = torch.tensor(trg_list)
        return src_tensor, trg_tensor

    def translate(self, sentence_tensor, vocab_type):
        # 在i2s词表中直接遍历查询即可
        sentence = ''
        for word_id_idx in range(len(sentence_tensor)):
            word_id = sentence_tensor[word_id_idx].item()
            sentence = sentence + ' ' + self.vocab_i2s[vocab_type][word_id]
        # 去除特殊词
        special_words = [BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD]
        for special_word in special_words:
            sentence.replace(special_word, '')
        return sentence