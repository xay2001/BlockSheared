from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import validate_model, load_model
from src.data_loader import get_loaders
from src.logger import Logger
import torch

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
def evaluate_model(config):
    logger = Logger(config['log_path'])
    # 打印模型加载开始
    print("开始加载模型...")
    # 加载模型和 tokenizer 的代码

    # 根据 config 选择加载本地模型或在线模型
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

    # 加载训练好的模型权重
    model = load_model(model, config['save_path'])

    # 打印模型加载完成
    print("模型加载完成！")

    # 打印数据集加载开始
    print("开始加载数据集...")

    _, val_enc = get_loaders(
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

    # 评估模型，计算损失、困惑度和准确率
    print("开始评估模型...")
    avg_loss, perplexity, accuracy = validate_model(model, val_enc, logger)
    logger.log(f"Evaluation complete. Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.4f}")
    print(f"评估完成。困惑度: {perplexity:.4f}, 准确率: {accuracy:.4f}")