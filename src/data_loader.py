import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset


def load_data(dataset_name, split, batch_size):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 加载指定的数据集
    dataset = load_dataset(dataset_name, split=split)

    # 定义数据集预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    # 对数据集进行tokenization
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 创建PyTorch DataLoader
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
