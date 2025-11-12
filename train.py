""" Transformer模型训练 """

import argparse
import os.path
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from spacy.compat import pickle
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformer.lr_scheduler import LRScheduler
from utils import Tokenizer, ModelLoader
from constants import PAD_WORD
from transformer.transformer import Transformer


def performance_str(tag, epoch_i, epochs, loss, ppl, accuracy, lr, duration):
    return (f'[{tag}] epoch: {epoch_i}/{epochs}, loss: {loss:.4f}, ppl: {ppl:.4f}, accuracy: {100*accuracy:.2f}%, '
            f'lr: {lr:.4f}, duration: {duration:.2f}s')


def train():
    # ==================== 解析命令行参数，获取参数配置 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-output_dir', default='outputs/train')
    parser.add_argument('-data_pkl', default='outputs/preprocess/preprocess_data.pkl')
    parser.add_argument('-model_dir', default='')

    parser.add_argument('-layer_num', type=int, default=6)
    parser.add_argument('-head_num', type=int, default=8)
    parser.add_argument('-word_vec', type=int, default=512)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-warmup_steps', type=int, default=4000)
    parser.add_argument('-lr', type=float, default=1e-3)

    parser.add_argument('-save_mode', choices=['best', 'all'], default='best')
    parser.add_argument('-no_label_smooth', action='store_true')

    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    if opt.model_dir and not os.path.exists(opt.model_dir):
        # 如果要训练一个已经加载一半的模型但是路径不存在，则报错
        raise ValueError(f'param model_dir:{opt.model_dir} does not exist')

    # ==================== 加载训练和验证数据集 ====================
    # 读取pkl文件中的数据集
    data = pickle.load(open(opt.data_pkl, 'rb'))
    vocab = data['vocab']

    # 将pkl中预处理的setting和词表属性添加到训练的setting变量中
    opt.max_seq_len = data['settings'].max_seq_len
    opt.src_pad_idx = vocab['src'].index(PAD_WORD)
    opt.trg_pad_idx = vocab['trg'].index(PAD_WORD)
    opt.src_vocab_size = len(vocab['src'])
    opt.trg_vocab_size = len(vocab['trg'])

    # 将单词根据词表转换为对应的ID，并转换成对应的tensor张量
    train_data, valid_data = data['train'], data['valid']
    tokenizer = Tokenizer(vocab)

    # 开始tokenizer训练数据
    print(f'[Info] Start to tokenize train data. Please waiting soon...')
    start_tokenizer_train = time.time()
    train_src_tensor, train_trg_tensor = tokenizer.tokenize(train_data)
    duration = time.time() - start_tokenizer_train
    print(f'[Info] Tokenizer train data finished with duration {duration:.2f}s')

    # 开始tokenizer验证数据
    print(f'[Info] Start to tokenize valid data. Please waiting soon...')
    start_tokenizer_valid = time.time()
    valid_src_tensor, valid_trg_tensor = tokenizer.tokenize(valid_data)
    duration = time.time() - start_tokenizer_valid
    print(f'[Info] Tokenizer valid data finished with duration {duration:.2f}s')

    # 封装成dataset类
    train_dataset = TensorDataset(train_src_tensor, train_trg_tensor)
    valid_dataset = TensorDataset(valid_src_tensor, valid_trg_tensor)

    # 构造dataloader类，方便后面训练和统计指标
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=opt.batch_size, shuffle=True)

    # ==================== 初始化模型，学习率优化器等 ====================
    if not opt.model_dir:
        # 如果模型路径为空，则重头开始训练模型
        transformer = Transformer(
            layer_num=opt.layer_num,
            head_num=opt.head_num,
            word_vec=opt.word_vec,
            d_ff=opt.d_ff,
            src_vocab_size=opt.src_vocab_size,
            trg_vocab_size=opt.trg_vocab_size,
            max_seq_size=opt.max_seq_len,
            src_pad_idx=opt.src_pad_idx,
            trg_pad_idx=opt.trg_pad_idx,
            dropout=opt.dropout
        )
    else:
        # 如果模型路径不为空，则加载该模型并继续训练
        checkpoint = torch.load(opt.model_dir)
        transformer = ModelLoader(checkpoint).load_exist_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LRScheduler(optimizer, warmup_steps=opt.warmup_steps, model_size=opt.word_vec)

    # 进入epoch开始迭代训练
    min_valid_loss = 1e9
    for epoch in range(opt.epoch):
        # ==================== 开始在训练集上训练 ====================
        transformer.train()
        start_train_time = time.time()
        epoch_i = epoch + 1
        desc = '    - (Training)    '
        running_loss = 0.0
        correct = 0
        step = 0
        total_words = 0
        correct, running_loss, step, total_words = train_epoch(correct, desc, device, opt, running_loss, scheduler,
                                                               step, total_words, train_dataloader, transformer)
        train_acc = correct / total_words
        train_loss = running_loss / step
        train_ppl = np.exp(min(train_loss, 100))
        train_duration = time.time() - start_train_time
        lr = scheduler.get_lr()
        # 将当前学习率记录到settings中，方便继续学习训练该模型
        opt.lr = lr
        train_performances = performance_str('Training', epoch_i, opt.epoch, train_loss, train_ppl, train_acc, lr, train_duration)
        print(train_performances)

        # ==================== 在验证集上统计指标 ====================
        transformer.eval()
        valid_acc, valid_loss, valid_performances, valid_ppl = valid_epoch(device, epoch_i, lr, opt,
                                                                               transformer, valid_dataloader)

        # ==================== 根据不同的保存策略保存模型 ====================
        # 模型参数，epoch_i和opt都需要保存
        checkpoint = {'params': transformer.state_dict(), 'epoch': epoch_i, 'settings': opt}
        if opt.save_mode == 'all':
            # 保存每个周期生成的checkpoint，
            torch.save(checkpoint, os.path.join(opt.output_dir, f'model_acc_{valid_acc*100:3.3f}.chkpt'))
        else:
            # 保存验证集损失值最低对应的模型
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(checkpoint, os.path.join(opt.output_dir, 'model.chkpt'))

        # ==================== 统计指标写入文件 ====================
        # 分别将统计指标写入到训练日志和验证日志中，方便观察训练结果
        train_log = os.path.join(opt.output_dir, 'train.log')
        valid_log = os.path.join(opt.output_dir, 'valid.log')
        with open(train_log, 'w', encoding='utf-8') as train_file, open(valid_log, 'w', encoding='utf-8') as valid_file:
            train_file.write(train_performances)
            valid_file.write(valid_performances)

        # ==================== tensorboard对训练过程可视化 ====================
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))
        tb_writer.add_scalars('accuracy', {'train': 100*train_acc, 'valid': 100*valid_acc}, epoch_i)
        tb_writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch_i)
        tb_writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch_i)
        tb_writer.add_scalars('ppl', {'train': train_ppl, 'valid': valid_ppl}, epoch_i)
        tb_writer.add_scalar('lr', lr, epoch_i)


def train_epoch(correct, desc, device, opt, running_loss, scheduler, step, total_words, train_dataloader, transformer):
    for src, trg in tqdm(train_dataloader, desc=desc, mininterval=2, leave=False):
        # 前向计算
        print('[Info] Start forward computing...')
        src.to(device)
        trg.to(device)
        pred = transformer(src, trg)
        print('[Info] Finish forward computing...')
        # 反向传播
        print('[Info] Start backward computing...')
        # 统计未 padding 的词
        cur_correct, loss, valid_words_num = cal_loss(opt, pred, trg)
        scheduler.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=5)

        scheduler.update_and_step()
        print('[Info] Finish backward computing...')
        # 指标统计
        running_loss += loss.item()
        step += 1
        correct += cur_correct
        total_words += valid_words_num
    return correct, running_loss, step, total_words


def valid_epoch(device, epoch_i, lr, opt, transformer, valid_dataloader):
    with torch.no_grad():
        start_valid_time = time.time()
        desc = '    - (Validating)    '
        running_loss = 0.0
        correct = 0
        step = 0
        total_words = 0
        for src, trg in tqdm(valid_dataloader, desc=desc, mininterval=2, leave=False):
            # 前向计算
            src.to(device)
            trg.to(device)
            pred = transformer(src, trg)
            # 指标统计
            cur_correct, loss, valid_words_num = cal_loss(opt, pred, trg)
            running_loss += loss.item()
            step += 1
            correct += cur_correct
            total_words += valid_words_num
        valid_acc = correct / total_words
        valid_loss = running_loss / step
        valid_ppl = np.exp(min(valid_loss, 100))
        valid_duration = time.time() - start_valid_time
        valid_performances = performance_str('Validating', epoch_i, opt.epoch, valid_loss, valid_ppl, valid_acc,
                                             lr, valid_duration)
        print(valid_performances)
    return valid_acc, valid_loss, valid_performances, valid_ppl


def cal_loss(opt, pred, trg):
    no_padding_mask = trg.ne(opt.trg_pad_idx)
    valid_words_num = no_padding_mask.sum()
    if opt.no_label_smooth:
        # 非标签平滑，标准的0-1编码
        # 将预测词和目标词映射成序列 (-1 * 词汇表数) 和 (-1)
        squeeze_pre = pred.view(-1, opt.trg_vocab_size)
        squeeze_trg = trg.view(-1)
        # 使用ignore_index忽略填充词，并使用均值作为汇聚函数
        loss = F.cross_entropy(squeeze_pre, squeeze_trg, ignore_index=opt.trg_pad_idx, reduction='mean')
        pred_arg_max = pred.argmax(dim=-1)
        cur_correct = pred_arg_max.eq(trg).sum().item()
    else:
        # 标签平滑，使用最小值避免极端的0-1场景出现
        loss, cur_correct = smooth_label_process(no_padding_mask, opt, pred, trg, valid_words_num)
    return cur_correct, loss, valid_words_num


def smooth_label_process(no_padding_mask, opt, pred, trg, valid_words_num):
    eps = 0.1
    # 建立idx索引序，举例而言trg_one_hot_idx中的[0, 1, 15]表示在one_hot_trg中的(0, 1, 15)处的值近似为1
    batch_idx = torch.arange(opt.batch_size).view(-1, 1, 1).expand(-1, opt.max_seq_len, 1)
    seq_idx = torch.arange(opt.max_seq_len).view(1, -1, 1).expand(opt.batch_size, -1, 1)
    vocab_idx = trg.unsqueeze(-1)
    trg_one_hot_idx = torch.cat([batch_idx, seq_idx, vocab_idx], dim=-1)
    # 构建平滑标签，使用极小值eps避免分布过于尖锐
    smooth_zero_value = eps / opt.trg_vocab_size
    smooth_one_value = 1 - eps + eps / opt.trg_vocab_size
    # dtype声明很关键！不然极小值会被转换成0
    one_hot_trg = torch.full((opt.batch_size, opt.max_seq_len, opt.trg_vocab_size), smooth_zero_value,
                             dtype=torch.float32)
    one_hot_trg.scatter_(2, trg_one_hot_idx, smooth_one_value)
    # 对pred进行log_softmax
    pred_log_softmax = F.log_softmax(pred, dim=-1)
    # per_sentence_loss 的格式为 (batch_size, max_seq_len)，交叉熵相当于是针对输出维度的最后一维(512)进行了汇聚计算
    per_sentence_loss = -(one_hot_trg * pred_log_softmax).sum(dim=-1)
    # 如果是填充词，不需要算在损失函数中
    total_loss = (per_sentence_loss * no_padding_mask.float()).sum()
    loss = total_loss / valid_words_num
    # 统计预测正确的单词数量
    one_hot_trg_argmax = one_hot_trg.argmax(dim=-1)
    pred_log_softmax_argmax = pred_log_softmax.argmax(dim=-1)
    cur_correct = one_hot_trg_argmax.eq(pred_log_softmax_argmax).sum().item()
    return loss, cur_correct


if __name__ == '__main__':
    train()