import torch
from datasets import load_dataset  # 使用datasets库加载C4数据集
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_data(dataset_name, split, batch_size):
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    # 加载 C4 数据集
    dataset = load_dataset(dataset_name, split=split)

    # 定义数据集预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    # 对数据集进行tokenization
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

    # 创建PyTorch DataLoader
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

