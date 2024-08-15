import torch
import time
import yaml
import argparse
from src.blockllm import BlockLLM
from src.data_loader import get_loaders
from src.utils import log_training_time, log_memory_usage, save_model
from src.logger import Logger
from transformers import AutoModelForCausalLM, AutoTokenizer  # 使用AutoModel和AutoTokenizer加载模型


def select_device(config):
    if config['use_gpu'] and torch.cuda.is_available():
        if isinstance(config['gpu_ids'], list) and len(config['gpu_ids']) > 0:
            device = torch.device(f"cuda:{config['gpu_ids'][0]}")  # 使用指定的第一个GPU
            print(f"检测到多GPU，使用GPU: {config['gpu_ids']} 号")
        else:
            device = torch.device("cuda")  # 使用默认的第一个可用GPU
            print(f"默认使用GPU: 0 号")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    return device

def fine_tune_model(config):
    logger = Logger(config['log_path'])
    # 打印模型加载开始
    print("开始加载模型...")

    if config['use_local_model']:
        model_path = config['local_model_path']
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model_name = config['hf_model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    device = select_device(config)  # 选择设备
    model.to(device)  # 将模型移动到设备

    if len(config['gpu_ids']) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=config['gpu_ids'])  # 使用多个GPU
        print(f"Using DataParallel on GPUs: {config['gpu_ids']}")

    print("模型加载完成并移动到设备！")

    # 打印数据集加载开始
    print("开始加载数据集...")

    # 使用 get_loaders 函数加载数据集
    train_loader, _ = get_loaders(
        config['dataset_name'],
        nsamples=config['batch_size'],
        seed=42,
        seqlen=2048,
        tokenizer=tokenizer,
        use_local=config['use_local_data'],
        local_paths=config['local_data_paths']
    )

    # 打印数据集加载完成
    print("数据集加载完成！")

    # 开始训练模型
    print("开始训练模型...")

    learning_rate = float(config['learning_rate'])
    block_llm = BlockLLM(model, config['sparsity'], config['patience'], learning_rate, logger)

    for epoch in range(config['epochs']):
        start_time = time.time()
        total_loss = 0
        for inp, tar in train_loader:
            inp, tar = inp.to(model.device), tar.to(model.device)
            loss = block_llm.train_step({'input': inp, 'label': tar})
            total_loss += loss

        epoch_time = time.time() - start_time
        log_training_time(epoch, epoch_time, logger)
        log_memory_usage(logger)

        logger.log(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
        print(f"第 {epoch + 1} 轮训练完成。平均损失: {total_loss / len(train_loader):.4f}")

    save_model(model, config['save_path'])
    logger.log(f"Model saved to {config['save_path']}")
    print(f"模型已保存到: {config['save_path']}")
    return model