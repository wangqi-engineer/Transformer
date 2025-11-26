# Transformer
使用Pytorch实现Transformer模型，在WMT数据集上实现英文到中文的文本翻译任务。

# 使用方法

## WMT-Chinese-to-English-Machine-Translation-Training-Corpus

## 0) 下载数据集

数据集（WMT-Chinese-to-English-Machine-Translation-Training-Corpus）源自世界翻译大会WMT 2021新闻翻译任务，整合了ParaCrawl、News-Commentary、Wiki-Titles、UN Parallel Corpus、WikiMatrix、CCMT等多个数据集，总计包含2500万对双语句子。
(http://www.statmt.org/wmt16/multimodal-task.html).

**将下载好的数据集放在和preprocess.py平级的data文件夹内**

```bash
# git lfs install
git clone https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus.git
```

## 1) 数据预处理
```bash
python preprocess.py -data_dir data -output_dir outputs/preprocess -save_data preprocess_data.pkl -max_seq_len 100 -sample_size 20000 -vocab_range train -vocab_min_freq 3
```

## 2) 模型训练
```bash
python train.py -b 50 -epoch 5 -output_dir outputs/train -data_pkl preprocess_data.pkl -save_mode best -monitor_steps 10 -on_win
```

## 3) 模型测试
```bash
python translate.py -b 128 -model_dir outputs/train/model.chkpt -output_dir outputs/translate
```

# 性能表现
## 训练

- 参数设置:
  - RTX 5080
  - sample size 20,000
  - batch size 50 
  - warmup step 4000 
  - epoch 5
  - label smoothing
  
## 测试 
- 2W数据集在5轮训练之后准确率可以达到 74%，训练耗时7min
---
# 下一步计划
  - 增加数据集数量，在百万规模级数据下训练模型的实际效果
  - 使用BPE分词器或者WordPiece分词器优化，而非简单选择词汇表并编码
  - 使用束采样或者温度+top p采样优化翻译效果
