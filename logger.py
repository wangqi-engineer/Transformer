""" 模型训练日志类 """
import logging
from pathlib import Path
from logging import Logger

__author__ = "Wang Qi"

class TransformerLogger(Logger):
    def __init__(self, name, level=logging.INFO, file=None):
        """ 设置日志的控制台和文件写入，方便观察性能指标等 """
        super().__init__(name, level)
        fmt = '[%(asctime)s][%(name)s][%(levelname)s][%(filename)s: %(lineno)d]: %(message)s'
        log_format = logging.Formatter(fmt=fmt)
        console_handle = logging.StreamHandler()
        console_handle.setFormatter(log_format)
        self.addHandler(console_handle)
        if file:
            file_handle = logging.FileHandler(file, encoding='utf-8')
            file_handle.setFormatter(log_format)
            self.addHandler(file_handle)

    @staticmethod
    def setup_logger(log_dir=None):
        name = 'transformer ' + Path(log_dir).stem if log_dir else ''
        return TransformerLogger(name, file=log_dir)