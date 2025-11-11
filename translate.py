""" Transformer模型翻译文本 """

import argparse
import os.path
import pickle
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from utils import Tokenizer, ModelLoader


def main():
    # ==================== 解析命令行参数，获取参数配置 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', default='outputs/train/model.chkpt')
    parser.add_argument('-output_dir', default='outputs/translate')
    parser.add_argument('-b', '--batch_size', type=int, default=128)

    opt = parser.parse_args()

    if not os.path.exists(opt.model_dir):
        raise ValueError('param model_dir does not exist')

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # ==================== 加载模型 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer = ModelLoader(opt.model_dir).load_exist_model()
    transformer.to(device)

    # ==================== 加载验证数据集 ====================
    # 读取文件中的数据集并tokenizer为张量
    data = pickle.load(open(opt.data_pkl, 'rb'))
    print(f'[Info] Start to tokenize test data. Please waiting soon...')
    start = time.time()
    tokenizer = Tokenizer(data['vocab'])
    test_src_tensor, test_trg_tensor = tokenizer.tokenize(data['test'])
    duration = time.time() - start
    print(f'[Info] Tokenizer test data finished with duration {duration:.2f}s')

    # 封装成dataset类
    test_dataset = TensorDataset(test_src_tensor, test_trg_tensor)

    # 构造dataloader类，方便后面训练和统计指标
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True)

    # ==================== 将翻译的原文本和目标文本写入文件 ====================
    desc = '    - (Translating) '
    for src, trg in tqdm(test_dataloader, desc=desc, mininterval=2, leave=True):
        src.to(device)
        trg.to(device)
        pred = transformer(src, trg)
        pred_argmax = torch.argmax(pred, dim=-1)
        output_file = os.path.join(opt.output_dir, 'translate.txt')
        with open(output_file, 'w', encoding='utf-8') as file:
            for idx in range(src.size(0)):
                # 根据word_id在词表中查找对应的词
                src_sentence = tokenizer.translate(src[idx], 'src')
                pred_sentence = tokenizer.translate(pred_argmax[idx], 'trg')
                trg_sentence = tokenizer.translate(trg[idx], 'trg')

                # 将要翻译的语句，翻译的语句和时间语句分为一组进行打印
                file.write(f'[SRC] {src_sentence}\n')
                print(f'[SRC] {src_sentence}')

                file.write(f'[PRD] {pred_sentence}\n')
                print(f'[PRD] {pred_sentence}')

                file.write(f'[TRG] {trg_sentence}\n')
                print(f'[TRG] {trg_sentence}')

                file.write('\n')
                print()


if __name__ == '__main__':
    main()