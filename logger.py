""" 模型训练日志类 """
import logging
import os.path
from logging import Logger


class TransformerLogger(Logger):
    def __init__(self, name, level=logging.INFO, file=None):
        """ 设置日志的控制台和文件写入，方便观察性能指标等 """
        super().__init__(name, level)
        fmt = '[%(asctime)s][%(name)s][%(levelname)s][%(filename)s: %(lineno)d]: %(message)s'
        log_format = logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
        console_handle = logging.StreamHandler()
        console_handle.setFormatter(log_format)
        self.addHandler(console_handle)
        if file:
            file_handle = logging.FileHandler(file, encoding='utf-8')
            file_handle.setFormatter(log_format)
            self.addHandler(file_handle)

    @staticmethod
    def setup_logger(log_dir=None):
        name = os.path.splitext(log_dir) if log_dir else 'transformer_log'
        return TransformerLogger(name, file=log_dir)