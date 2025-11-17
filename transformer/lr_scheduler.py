""" Transformer模型学习率调整器 """
import numpy as np
from torch.cuda.amp import GradScaler


class LRScheduler:
    def __init__(self, optimizer, warmup_steps, model_size):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self._cur_step = 0
        # 开启混合精度
        self.scaler = GradScaler()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _update_lr(self):
        factor = np.power(self.model_size, -0.5)
        # 预热阶段学习率线性增加，避免Adam调整器导致的学习曲线震荡
        warmup_factor = self._cur_step * np.power(self.warmup_steps, -1.5)
        # 衰减阶段倒数平方根递减，快速衰减学习率找到局部最优解
        decay_factor = np.power(self._cur_step, -0.5)
        lr = factor * min(warmup_factor, decay_factor)
        # 学习率调整
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update_and_step(self):
        self._cur_step += 1
        self._update_lr()
        # 开启混合精度计算
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def backward(self, loss):
        # 开启混合精度进行反向传播
        self.scaler.scale(loss).backward()