""" 回滚函数，删除所output输出产物 """
import os.path
import shutil

if __name__ == '__main__':
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')