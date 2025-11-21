""" 数据准备和预处理 """
import argparse
import csv
import logging
import os.path
import pickle
import random
import time
from collections import Counter
from itertools import islice
from pathlib import Path

from constants import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD
from logger import TransformerLogger

log = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, examples, max_seq_len=100):
        self.examples = examples
        self.max_seq_len = max_seq_len

    def build_vocabulary(self, min_freq=1):
        # 初始化Counter计数器和预处理后的数据
        special_words = [PAD_WORD, BOS_WORD, EOS_WORD, UNK_WORD]
        src_counter, trg_counter = Counter(special_words), Counter(special_words)
        for text in self.examples:
            src_text_counter = Counter((text['src'].split()))
            trg_text_counter = Counter((text['trg'].split()))
            src_counter.update(src_text_counter), trg_counter.update(trg_text_counter)
        # 出现频率过低的词不加入词表中，特殊词需要登记到词表中
        src_vocab = [key for key, value in src_counter.items() if value >= min_freq or key in special_words]
        trg_vocab = [key for key, value in trg_counter.items() if value >= min_freq or key in special_words]
        return {'src': src_vocab, 'trg': trg_vocab}


class DataPreprocessor:
    """
    数据预处理加载器，作用是读取cvs文件数据，划分数据集，构建词表和添加特殊词的句子，然后保存为文件
    """
    def __init__(self, data_dir, max_seq_len=100):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

    def _preprocess_sentence(self, sentence):
        # ==================== 句子过长则截断 ====================
        # 构建词表时大小写不敏感
        words = sentence.lower().split()
        words_num = len(words)
        # 最大序列长度包含起始词和结束词，因此原始句子的最大序列长度为 max_seq_len 减去2
        words_max_len = self.max_seq_len - 2
        if words_num > words_max_len:
            sentence_clip = words[:words_max_len]
            sentence = " ".join(sentence_clip)

        # ==================== 添加起始词，结束词和填充词 ====================
        new_sentence = " ".join([BOS_WORD, sentence, EOS_WORD])
        # 填充的数量根据原始单词允许的最大序列长度减去当前已有的单词数去填充
        padding_num = words_max_len - words_num
        new_sentence = new_sentence + " " + " ".join(PAD_WORD for _ in range(padding_num))
        return new_sentence

    def load_raw_file2_pickle_file(self, opt):
        # ==================== 读取csv文件中的数据 ====================
        # 获取数据文件夹下的唯一一个csv文件
        data_dir_path = Path(self.data_dir)
        csv_files = list(data_dir_path.glob('*.csv'))
        if len(csv_files) < 0:
            raise ValueError(f'data dir path: {data_dir_path} does not contain csv file')

        # 读取csv文件
        with open(csv_files[0], 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            data = []
            start_time = time.time()
            sliced_reader = islice(csv_reader, int(opt.sample_size))
            log.info('Start to load data from csv. Please waiting soon...')
            for idx, row in enumerate(sliced_reader, start=1):
                # 读取数据集时就添加特殊词，完成填充/截断逻辑
                src_sentence = self._preprocess_sentence(row['1'])
                trg_sentence = self._preprocess_sentence(row['0'])
                data.append({str('src'): src_sentence, str('trg'): trg_sentence})
            duration = time.time() - start_time
            log.info(f'Load data from csv file finished with duration: {duration:.2f}s')

            # ==================== 将数据集划分为训练、验证和测试集 ====================
            shuffle_data = data.copy()
            random.shuffle(shuffle_data)
            data_size = len(data)
            # 按照8:1:1的比例划分训练集、验证集和测试集
            split1 = int(data_size * 0.8)
            split2 = split1 + int(data_size * 0.1)
            train_data, valid_data, test_data = shuffle_data[:split1], shuffle_data[split1:split2], shuffle_data[split2:]

            # ==================== 构建词表 ====================
            # 包含训练数据集的src和trg词表，已经添加特殊词和padding/slicing之后的原始文本数据
            log.info('Start to build vocabulary. Please waiting soon...')
            start_build_vocab = time.time()
            vocab_data = train_data if opt.vocab_range == 'train' else data
            train_vocab = Vocabulary(vocab_data, opt.max_seq_len).build_vocabulary(min_freq=opt.vocab_min_freq)
            duration = time.time() - start_build_vocab
            log.info(f'Build vocabulary finished with duration: {duration:.2f}s')

            # ==================== 数据存为文件 ====================
            # 将opt，词汇表和训练和验证的数据集全部存为文件
            data = {
                'settings': opt,
                'vocab': train_vocab,
                'train': train_data,
                'valid': valid_data,
                'test': test_data
            }
            pickle.dump(data, open(os.path.join(opt.output_dir, opt.save_data), 'wb'))
            log.info(f'Save vocabulary and data finished')


def main():
    """
    准备数据集，做好词汇表，完成数据集划分，特殊词处理等数据清洗工作
    使用：python preprocess.py -data_dir data -output_dir outputs/preprocess -save_data preprocess_data.pkl -max_seq_len 100 -sample_size 1e6 -vocab_range train -vocab_min_freq 3
    """

    # ==================== 解析命令行参数，获取参数配置 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='data', help='预处理的数据路径')
    parser.add_argument('-output_dir', default='outputs/preprocess', help='预处理输出路径')
    parser.add_argument('-save_data', default='preprocess_data.pkl', help='保存的词表、设置参数和数据集信息')
    parser.add_argument('-max_seq_len', type=int, default=100, help='最大序列长度')
    parser.add_argument('-sample_size', type=int, default=2e6, help='样本最大数量')

    parser.add_argument('-vocab_range', choices=['train', 'all'], default='train', help='词表范围，训练集还是全量样本')
    parser.add_argument('-vocab_min_freq', type=int, default=3, help='词表最小频率')

    opt = parser.parse_args()

    # 如果不存在数据集文件夹则创建
    if not os.path.exists(opt.data_dir):
        os.mkdir(opt.data_dir)

    # 如果输出文件夹不存在则创建
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    # 初始化日志信息
    log_dir = os.path.join(opt.output_dir, 'preprocess.log')
    global log
    log = TransformerLogger.setup_logger(log_dir)

    data_preprocessor = DataPreprocessor(opt.data_dir, opt.max_seq_len)
    # ==================== 数据预处理并通过文件保存 ====================
    data_preprocessor.load_raw_file2_pickle_file(opt)
    log.info('Preprocess dataset finish')


if __name__ == '__main__':
    main()