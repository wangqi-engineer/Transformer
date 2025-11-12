""" 抽取小样本数据，方便实机训练 """

import random
import csv


def random_sample_csv(input_file, output_file, sample_size=10000000):
    """
    使用水库抽样算法随机采样，适合超大文件
    """
    # 首先读取标题
    with open(input_file, 'r', encoding='utf-8') as f:
        header = f.readline()

    # 获取总行数（不包括标题）
    with open(input_file, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1

    if sample_size > total_rows:
        sample_size = total_rows

    # 使用水库抽样算法
    selected_indices = set(random.sample(range(total_rows), sample_size))

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)

            # 写入标题
            writer.writerow(next(reader))

            # 选择指定行
            for i, row in enumerate(reader):
                if i in selected_indices:
                    writer.writerow(row)

    print(f"采样完成，共 {sample_size} 行")

if __name__ == '__main__':
    # 定义输入和输出函数
    input_file, output_file = 'data/wmt_zh_en_training_corpus.csv', 'data/wmt_zh_en_training_corpus_tiny.csv'
    # 使用函数
    random_sample_csv(input_file, output_file, 10000000)