""" 工具类 """
import os
from sys import platform

import psutil
import torch
from tqdm import tqdm

from constants import UNK_WORD, BOS_WORD, EOS_WORD, PAD_WORD
from transformer.transformer import Transformer
import GPUtil


class Tokenizer:
    """ 将单词映射为id并转为tensor """
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
        for sentence in tqdm(data, desc=desc, mininterval=2, leave=False):
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

class ModelLoader:
    """ Transformer模型加载器 """
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def load_exist_model(self):
        settings = self.checkpoint['settings']

        transformer = Transformer(
            layer_num=settings.layer_num,
            head_num=settings.head_num,
            word_vec=settings.word_vec,
            d_ff=settings.d_ff,
            src_vocab_size=settings.src_vocab_size,
            trg_vocab_size=settings.trg_vocab_size,
            max_seq_size=settings.max_seq_len,
            src_pad_idx=settings.src_pad_idx,
            trg_pad_idx=settings.trg_pad_idx,
            dropout=settings.dropout
        )
        transformer.load_state_dict(self.checkpoint['params'])
        return transformer

class DeviceMonitor:
    """ 设备信息监控类 """
    def __init__(self, log):
        self.log = log

    def display_device_info(self):
        """ 打印训练设备详细信息 """
        self.log.info('=' * 60)
        self.log.info('[Device Info]')

        # GPU信息
        if torch.cuda.is_available():
            self.log.info(f'CUDA Available: {torch.cuda.is_available()}')
            self.log.info(f'CUDA Version: {torch.version.cuda}')
            self.log.info(f'Current GPU: {torch.cuda.current_device()}')
            self.log.info(f'GPU Name: {torch.cuda.get_device_name()}')
            self.log.info(f'GPU Count: {torch.cuda.device_count()}')

            # 详细GPU信息
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                self.log.info(f' GPU {i} Detail Info:')
                self.log.info(f'  ├─ ID: {gpu.id}')
                self.log.info(f"  ├─ Name: {gpu.name}")
                self.log.info(f"  ├─ Memory: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")
                self.log.info(f"  ├─ Memory Rate: {gpu.memoryUtil * 100:.1f}%")
                self.log.info(f"  ├─ Driver: {gpu.driver}")
                self.log.info(f"  └─ UUID: {gpu.uuid}")
        else:
            self.log.info('Cuda Unavailable, Use CPU Instead')

        # CPU信息
        self.log.info(f" CPU Info:")
        self.log.info(f"  ├─ Physical CPU Count: {psutil.cpu_count(logical=False)}")
        self.log.info(f"  ├─ Logical CPU Count: {psutil.cpu_count(logical=True)}")
        self.log.info(f"  ├─ CPU Frequency: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz")
        self.log.info(f"  └─ Architecture: {platform.processor() if hasattr(platform, 'processor') else 'N/A'}")

        # 内存信息
        memory = psutil.virtual_memory()
        self.log.info(f" Memory Info:")
        self.log.info(f"  ├─ Total Memory: {memory.total / (1024 ** 3):.1f} GB")
        self.log.info(f"  ├─ Available Memory: {memory.available / (1024 ** 3):.1f} GB")
        self.log.info(f"  └─ Percent Memory: {memory.percent}%")

        # PyTorch信息
        self.log.info(f" PyTorch Info:")
        self.log.info(f"  └─ Version: {torch.__version__}")

        self.log.info("=" * 60)

    def display_gpu_memory(self):
        if torch.cuda.is_available():
            self.log.info('=' * 60)
            self.log.info('[GPU Memory Usage]')

            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.log.info(f"GPU Memory: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")
                self.log.info(f"GPU Memory Rate: {gpu.memoryUtil * 100:.1f}%")


class ModelSizeEval:
    """ 评估模型大小 """
    def __init__(self, transformer, model_dir=None):
        if transformer is None:
            if not os.path.exists(model_dir):
                raise ValueError('param model_dir does not exist')

            # ==================== 加载模型 ====================
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_dir)
            transformer = ModelLoader(checkpoint).load_exist_model()
            transformer.to(device)
        self.transformer = transformer

    def calculate_model_size(self):
        # ==================== 评估模型大小 ====================
        total_params = sum(p.numel() for p in self.transformer.parameters())
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)

        status = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6
        }

        return status