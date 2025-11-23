""" Transformer模型训练 """

import argparse
import logging
import os.path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn
from spacy.compat import pickle
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast

from dataclass import TrainingStatics, TrainingTools
from logger import TransformerLogger
from transformer.lr_scheduler import SchedulerOptim
from utils import Tokenizer, ModelLoader, DeviceMonitor, ModelSizeEval
from constants import PAD_WORD
from transformer.transformer import Transformer

__author__ = "Wang Qi"

# 设置全局日志变量
log = logging.getLogger(__name__)

# 启用性能优化
torch.set_float32_matmul_precision('high')

# 设置最小验证集损失值
min_valid_loss = 1e9

# 设置当前步数和每轮的总步数
step = 0
total_step = 0

def performance_str(tag, epoch_i, epochs, loss, ppl, accuracy, lr, duration):
    return (f'[{tag}] training step: {step}/{total_step}, epoch: {epoch_i}/{epochs}, loss: {loss:.4f}, ppl: {ppl:.4f}, accuracy: {100 * accuracy:.2f}%, '
            f'lr: {lr:.4f}, duration: {duration:.2f}s')


def train():
    # ==================== 解析命令行参数，获取参数配置 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='每次从DataLoader中拿到的批数据大小')
    parser.add_argument('-epoch', type=int, default=10, help='训练轮数')
    parser.add_argument('-monitor_steps', type=int, default=100, help='每隔多少步检测一次显存/梯度流等信息')
    # parser.add_argument('-model_eval_steps', type=int, default=500, help='每隔多少步评估一次模型')
    # parser.add_argument('-model_save_steps', type=int, default=1000, help='每隔多少步保存一次模型')

    parser.add_argument('-output_dir', default='outputs/train', help='输出路径')
    parser.add_argument('-data_pkl', default='outputs/preprocess/preprocess_data.pkl', help='预处理阶段保存的词表、设置参数和数据集信息')
    parser.add_argument('-model_dir', default='', help='模型路径，如果初次训练为空；如果训练已有模型填写对应路径')

    parser.add_argument('-layer_num', type=int, default=6, help='编码器和解码器的层数')
    parser.add_argument('-head_num', type=int, default=8, help='多头注意力机制的头数')
    parser.add_argument('-word_vec', type=int, default=512, help='Embedding维度')
    parser.add_argument('-d_ff', type=int, default=2048, help='feed back层高频映射维度')
    parser.add_argument('-dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('-warmup_steps', type=int, default=4000, help='预热步数')
    parser.add_argument('-lr_mul', type=float, default=2.0, help='学习率伸缩因子')

    parser.add_argument('-save_mode', choices=['best', 'all'], default='best', help='保存模型方式；全量保存还是最有保存')
    parser.add_argument('-no_label_smooth', action='store_true', help='是否设置标签平滑')

    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    # 初始化日志信息
    log_dir = os.path.join(opt.output_dir, 'train.log')
    global log
    log = TransformerLogger.setup_logger(log_dir)

    if opt.model_dir and not os.path.exists(opt.model_dir):
        # 如果要训练一个已经加载一半的模型但是路径不存在，则报错
        err_msg = f'param model_dir:{opt.model_dir} does not exist'
        log.error(err_msg)
        raise ValueError(err_msg)

    # if opt.model_save_steps % opt.model_eval_steps != 0:
    #     # 保存步数需要为验证步数的整数倍，不然保存会失败
    #     err_msg = f'model_save_steps:{opt.model_save_steps} must be an integer multiple of model_eval_steps: {opt.model_eval_steps}'
    #     log.error(err_msg)
    #     raise ValueError(err_msg)

    # ==================== 初始化训练监控类 ====================
    device_monitor = DeviceMonitor(log)
    device_monitor.display_device_info()

    # ==================== 打印配置信息 ====================
    log.info(f'[Settings] Settings: {opt}')
    log.info('=' * 60)

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
    log.info(f'[Data Size] Train data size: {len(train_data)}')
    log.info(f'[Data Size] Valid data size: {len(valid_data)}')
    log.info('=' * 60)
    global total_step
    total_step = int(len(train_data) / opt.batch_size)

    # 开始tokenizer训练数据
    log.debug(f'Start to tokenize train data. Please waiting soon...')
    start_tokenizer_train = time.time()
    train_src_tensor, train_trg_tensor = tokenizer.tokenize(train_data)
    duration = time.time() - start_tokenizer_train
    log.debug(f'Tokenizer train data finished with duration {duration:.2f}s')

    # 开始tokenizer验证数据
    log.debug(f'Start to tokenize valid data. Please waiting soon...')
    start_tokenizer_valid = time.time()
    valid_src_tensor, valid_trg_tensor = tokenizer.tokenize(valid_data)
    duration = time.time() - start_tokenizer_valid
    log.debug(f'Tokenizer valid data finished with duration {duration:.2f}s')

    # 封装成dataset类
    train_dataset = TensorDataset(train_src_tensor, train_trg_tensor)
    valid_dataset = TensorDataset(valid_src_tensor, valid_trg_tensor)

    # 构造dataloader类，方便后面训练和统计指标

    #   采用多线程加载数据，开启线程预加载，开启锁页内存，确保加载线程存活
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True
    )

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
    # 使用pytorch2.0编译优化加速训练
    if hasattr(torch, 'compile'):
        transformer = torch.compile(transformer)
    else:
        log.warning('NO PYTORCH 2.0 COMPILATION OPTIMIZATION')
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = SchedulerOptim(optimizer, warmup_steps=opt.warmup_steps, model_size=opt.word_vec, lr_mul=opt.lr_mul)

    # 模型大小
    model_status = ModelSizeEval(transformer).calculate_model_size()
    log.info(f"[Model Size] Total model parameters size: {model_status['total_M']:.1f}M")
    log.info(f"[Model Size] Trainable model parameters size: {model_status['trainable_M']:.1f}M")
    log.info('=' * 60)

    # 模型初始化
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Embedding):
            # Embedding层：使用较小的初始化
            nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim ** -0.5)
            log.info(f"    初始化Embedding: {name}")

        elif isinstance(module, nn.Linear):
            # 根据层类型选择不同的初始化策略
            if 'w_q' in name or 'w_k' in name:
                # Q、K矩阵：极小的初始化防止softmax饱和
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                log.info(f"     Init Q/K: {name} (gain=0.1)")

            elif 'w_v' in name:
                # V矩阵：中等初始化
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                log.info(f"     Init V: {name} (gain=0.5)")

            elif 'output' in name or 'generator' in name or 'projection' in name:
                # 输出层：较小的初始化
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                log.info(f"     Init output: {name} (gain=0.3)")

            elif 'feed_forward' in name or 'ffn' in name:
                # FFN层：正常初始化
                if '.0' in name or 'w_1' in name:  # 第一个线性层
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:  # 第二个线性层
                    nn.init.xavier_uniform_(module.weight, gain=0.3)
                log.info(f"     Init FFN: {name}")

            else:
                # 其他线性层：安全初始化
                nn.init.xavier_uniform_(module.weight, gain=0.7)
                log.info(f"     Init Linear: {name} (gain=0.7)")

            # 偏置初始化
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm：标准初始化
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # 进入epoch开始迭代训练
    for epoch in range(opt.epoch):
        # ==================== 开始在训练集上训练 ====================
        start_train_time = time.time()
        epoch_i = epoch + 1
        running_loss = 0.0
        correct = 0
        global step
        step = 0
        total_words = 0

        training_statics_epoch = TrainingStatics(
            correct=correct,
            epoch_i=epoch_i,
            running_loss=running_loss,
            total_words=total_words
        )

        training_tools_epoch = TrainingTools(
            device=device,
            device_monitor=device_monitor,
            opt=opt,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            transformer=transformer
        )

        correct, running_loss, total_words = train_epoch(training_statics_epoch, training_tools_epoch)

        train_acc = correct / total_words
        train_loss = running_loss / step
        train_ppl = np.exp(min(train_loss, 100))
        train_duration = time.time() - start_train_time
        log.info(f'[TRAINING EPOCH] epoch {epoch_i}/{opt.epoch} has been finished')

        training_statics = TrainingStatics(
            train_acc=train_acc,
            train_duration=train_duration,
            train_loss=train_loss,
            train_ppl=train_ppl,
            epoch_i=epoch_i,
        )

        training_tools = TrainingTools(
            device=device,
            scheduler=scheduler,
            opt=opt,
            transformer=transformer,
            valid_dataloader=valid_dataloader,
            device_monitor=device_monitor
        )

        record_status(training_statics, training_tools, epoch_finish=True)


def train_epoch(training_statics_epoch: TrainingStatics, training_tools_epoch: TrainingTools):
    desc = '    - (Training)    '
    gradient_history = []
    training_tools_epoch.transformer.train()

    # 记录每层前向输出
    activations = {}
    gradient_norms = {}

    def forward_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradient_norms[name] = grad_output[0].norm().item()

        return hook

    # 一次性注册所有hook
    forward_hooks = []
    backward_hooks = []

    for name, module in training_tools_epoch.transformer.named_modules():
        if len(list(module.children())) == 0:  # 只注册叶子模块
            forward_hook_handle = module.register_forward_hook(forward_hook(name))
            backward_hook_handle = module.register_full_backward_hook(backward_hook(name))
            forward_hooks.append(forward_hook_handle)
            backward_hooks.append(backward_hook_handle)

    for src, trg in tqdm(training_tools_epoch.train_dataloader, desc=desc, mininterval=2, leave=False):
        global step
        step += 1

        # 记录训练前的模型参数，检查模型是否更新
        initial_params = [p.clone() for p in training_tools_epoch.transformer.parameters()]
        train_step_start = time.time()

        # 清空上一轮的统计
        activations.clear()
        gradient_norms.clear()

        # 梯度清零
        training_tools_epoch.scheduler.zero_grad()

        # 前向计算
        log.debug('Start forward computing...')
        src = src.to(training_tools_epoch.device)
        trg = trg.to(training_tools_epoch.device)

        # 开启混合精度
        with autocast('cuda'):
            pred = training_tools_epoch.transformer(src, trg)
            log.debug('Finish forward computing...')

            # 统计未 padding 的词
            cur_correct, loss, valid_words_num = cal_loss(training_tools_epoch.opt, pred, trg)

        # 反向传播
        log.debug('Start backward computing...')
        training_tools_epoch.scheduler.backward(loss)

        # 在混合精度中unscale梯度以便正确统计和裁剪
        training_tools_epoch.scheduler.scaler.unscale_(training_tools_epoch.scheduler.optimizer)

        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(training_tools_epoch.transformer.parameters(), max_norm=5.0)

        # 计算梯度统计
        grad_norms = {}
        total_norm = 0

        for name, param in training_tools_epoch.transformer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                total_norm += grad_norm

        # 分析每层
        if step % training_tools_epoch.opt.monitor_steps == 0:
            log.info('[GARD ANALISE]')
            log.info("LAYER NAME | FORWARD RANGE | BACKWARD GRAD | STATUS")
            log.info("-" * 60)

            for name, module in training_tools_epoch.transformer.named_modules():
                if name in activations and name in gradient_norms:
                    act = activations[name]
                    grad_norm = gradient_norms[name]

                    status = "NORMAL" if grad_norm > 1e-6 else "SMALL" if grad_norm > 1e-9 else "VANISHED"
                    log.debug(f"{name:30} | [{act.min():.4f}, {act.max():.4f}] | {grad_norm:.2e} | {status}")
                    if status != "NORMAL":
                        log.warning(f"{name:30} | [{act.min():.4f}, {act.max():.4f}] | {grad_norm:.2e} | {status}")

                    if status == "VANISHED":
                        log.error(f"{name:30} | [{act.min():.4f}, {act.max():.4f}] | {grad_norm:.2e} | {status}")
                        # 尝试实时修复消失的梯度
                        if 'w_q' in name or 'w_k' in name or 'w_v' in name:
                            for param_name, param in module.named_parameters():
                                if param.grad is not None and param.grad.norm().item() < 1e-10:
                                    # 注入人工梯度
                                    with torch.no_grad():
                                        pseudo_grad = torch.randn_like(param) * 1e-8
                                        param.grad += pseudo_grad
                                    log.info(f"Param {param_name} inject artificial gradients")

        gradient_history.append({
            'step': step,
            'epoch': training_statics_epoch.epoch_i,
            'loss': loss.item(),
            'total_grad_norm': total_norm,
            'layer_grads': grad_norms
        })

        training_tools_epoch.scheduler.update_and_step()
        log.debug('Finish backward computing...')

        # 确保CUDA操作完成
        if training_tools_epoch.device.type == 'cuda':
            torch.cuda.synchronize()

        # 检查参数是否变化
        param_changed = False
        max_change = 0.0
        for i, (init_param, current_param) in enumerate(
                zip(initial_params, training_tools_epoch.transformer.parameters())):
            change = (current_param - init_param).abs().max().item()
            max_change = max(max_change, change)

            # 使用更宽松的容差，特别是对于混合精度
            if change > 1e-8:  # 只要变化大于1e-8就认为更新了
                param_changed = True
                log.debug(f"Model param {i} updated, max change: {change:.2e}")
                break

        if not param_changed:
            log.warning(f"Model param has not been updated! Max change: {max_change:.2e}")
        else:
            log.debug(f"Parameters updated successfully, max change: {max_change:.2e}")

        lr = training_tools_epoch.scheduler.get_lr()
        if step % training_tools_epoch.opt.monitor_steps == 0:
            # 检测当前设备gpu显存的使用情况
            training_tools_epoch.device_monitor.display_gpu_memory(step, total_step, training_statics_epoch.epoch_i,
                                                             training_tools_epoch.opt.epoch)

            # 检查梯度变化情况
            log.info(f"[GRAD CHECKPOINT] epoch {training_statics_epoch.epoch_i}, step {step}, loss={loss.item():.4f}, "
                     f"total grad norm={total_norm:.6f}, lr={lr}")

        # 检测梯度消失
        if total_norm < 1e-10:
            log.warning(f"[GRADIENT VANISHING] Gradient {total_norm:.6f} vanishing occurs in epoch {training_statics_epoch.epoch_i}, "
                        f"epoch {training_statics_epoch.epoch_i}, step {step}, lr={lr}")

        # 指标统计
        training_statics_epoch.running_loss += loss.item()
        training_statics_epoch.correct += cur_correct
        training_statics_epoch.total_words += valid_words_num

        # train_loss = loss.item()
        # train_acc = cur_correct / valid_words_num
        # train_ppl = np.exp(min(train_loss, 100))
        # train_duration = time.time() - train_step_start
        #
        # training_statics = TrainingStatics(
        #     train_acc=train_acc,
        #     train_duration=train_duration,
        #     train_loss=train_loss,
        #     train_ppl=train_ppl,
        #     epoch_i=training_statics_epoch.epoch_i,
        # )
        #
        # training_tools = TrainingTools(
        #     device=training_tools_epoch.device,
        #     scheduler=training_tools_epoch.scheduler,
        #     opt=training_tools_epoch.opt,
        #     transformer=training_tools_epoch.transformer,
        #     valid_dataloader=training_tools_epoch.valid_dataloader,
        #     device_monitor=training_tools_epoch.device_monitor
        # )
        #
        # record_status(training_statics, training_tools, epoch_finish=False)
    return training_statics_epoch.correct, training_statics_epoch.running_loss, training_statics_epoch.total_words


def record_status(training_statics: TrainingStatics, training_tools: TrainingTools, epoch_finish=False):
    # 每训练若干步或者一轮训练结束后记录指标并保存模型
    # if epoch_finish or step != total_step and step % training_tools.opt.model_eval_steps == 0:
    if epoch_finish:
        lr = training_tools.scheduler.get_lr()
        # 将当前学习率记录到settings中，方便继续学习训练该模型
        training_tools.opt.lr = lr
        train_performances = performance_str('Training', training_statics.epoch_i, training_tools.opt.epoch,
                                             training_statics.train_loss, training_statics.train_ppl,
                                             training_statics.train_acc, lr, training_statics.train_duration)
        log.info(train_performances)
        # ==================== 在验证集上统计指标 ====================
        training_tools.transformer.eval()
        valid_acc, valid_loss, valid_performances, valid_ppl = valid_epoch(training_tools.device, training_statics.epoch_i,
                                                                           lr, training_tools.opt, training_tools.transformer,
                                                                           training_tools.valid_dataloader)

        # if epoch_finish or step != total_step and step % training_tools.opt.model_save_steps == 0:
        # ==================== 根据不同的保存策略保存模型 ====================
        # 模型参数，epoch_i和opt都需要保存
        checkpoint = {'params': training_tools.transformer.state_dict(),
                      'epoch': training_statics.epoch_i,
                      'step': step,
                      'settings': training_tools.opt}
        if training_tools.opt.save_mode == 'all':
            # 保存每个周期生成的checkpoint，
            torch.save(checkpoint, os.path.join(training_tools.opt.output_dir, f'model_acc_{valid_acc * 100:3.3f}.chkpt'))
        else:
            # 保存验证集损失值最低对应的模型
            global min_valid_loss
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(checkpoint, os.path.join(training_tools.opt.output_dir, 'model.chkpt'))

        # ==================== 统计指标写入文件 ====================
        # 分别将统计指标写入到训练日志和验证日志中，方便观察训练结果
        train_log = os.path.join(training_tools.opt.output_dir, 'train_perf.log')
        valid_log = os.path.join(training_tools.opt.output_dir, 'valid_perf.log')
        with open(train_log, 'a', encoding='utf-8') as train_file, open(valid_log, 'a', encoding='utf-8') as valid_file:
            train_file.write(train_performances + '\n')
            valid_file.write(valid_performances + '\n')

        # ==================== tensorboard对训练过程可视化 ====================
        tb_writer = SummaryWriter(log_dir=os.path.join(training_tools.opt.output_dir, 'tensorboard'))
        tb_writer.add_scalars('accuracy',
                              {'train': 100 * training_statics.train_acc, 'valid': 100 * valid_acc},
                              step)
        tb_writer.add_scalars('loss',
                              {'train': training_statics.train_loss, 'valid': valid_loss},
                              step)
        tb_writer.add_scalars('loss',
                              {'train': training_statics.train_loss, 'valid': valid_loss},
                              step)
        tb_writer.add_scalars('ppl',
                              {'train': training_statics.train_ppl, 'valid': valid_ppl},
                              step)
        tb_writer.add_scalar('lr', lr, step)


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
            src = src.to(device)
            trg = trg.to(device)
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
        log.info(valid_performances)
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
    cur_batch_size = trg.size(0)
    # 建立idx索引序，举例而言trg_one_hot_idx中的[0, 1, 15]表示在one_hot_trg中的(0, 1, 15)处的值近似为1
    batch_idx = torch.arange(cur_batch_size).view(-1, 1, 1).expand(-1, opt.max_seq_len, 1).to(trg.device)
    seq_idx = torch.arange(opt.max_seq_len).view(1, -1, 1).expand(cur_batch_size, -1, 1).to(trg.device)
    vocab_idx = trg.unsqueeze(-1)
    trg_one_hot_idx = torch.cat([batch_idx, seq_idx, vocab_idx], dim=-1)
    # 构建平滑标签，使用极小值eps避免分布过于尖锐
    smooth_zero_value = eps / opt.trg_vocab_size
    smooth_one_value = 1 - eps + eps / opt.trg_vocab_size
    # dtype声明很关键！不然极小值会被转换成0
    one_hot_trg = torch.full((cur_batch_size, opt.max_seq_len, opt.trg_vocab_size), smooth_zero_value,
                             dtype=torch.float32, device=trg.device)
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
    try:
        train()
    except Exception as e:
        # 获取完整的堆栈信息
        stack_trace = traceback.format_exc()

        # 记录详细错误信息
        log.error("=" * 60)
        log.error(f"[ERROR TYPE] {type(e).__name__}")
        log.error(f"[ERROR MSG] {str(e)}")
        log.error(f"[STACK TRACE]\n{stack_trace}" )
        log.error("=" * 60)
