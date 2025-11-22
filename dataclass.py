""" 定义类装饰器用于代码重构 """
from dataclasses import dataclass
from typing import Any

import torch

@dataclass
class TrainingStatics:
    """ 训练的指标数据 """
    train_acc: float = 0.0
    train_duration: float = 0.0
    train_loss: float = 0.0
    train_ppl: float = 0.0
    epoch_i: int = 1
    correct: int = 0
    running_loss: float = 0
    step: int = 0
    total_words: int = 0


@dataclass
class TrainingTools:
    """ 训练工具 """
    device: torch.device = None
    scheduler: Any = None
    opt: Any = None
    transformer: Any = None
    valid_dataloader: Any = None
    train_dataloader: Any = None
    device_monitor: Any = None