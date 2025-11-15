""" Transformer模型参数评估 """

import argparse
import os.path
import torch

from utils import ModelLoader


def main():
    # ==================== 解析命令行参数，获取参数配置 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', default='outputs/train/model.chkpt')

    opt = parser.parse_args()

    if not os.path.exists(opt.model_dir):
        raise ValueError('param model_dir does not exist')

    # ==================== 加载模型 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(opt.model_dir)
    transformer = ModelLoader(checkpoint).load_exist_model()
    transformer.to(device)

    # ==================== 评估模型大小 ====================
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

    status = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_M': total_params / 1e6,
        'trainable_M': trainable_params / 1e6
    }

    print(f"总参数量: {status['total_M']:.1f}M")
    print(f"可训练参数量: {status['trainable_M']:.1f}M")


if __name__ == '__main__':
    main()