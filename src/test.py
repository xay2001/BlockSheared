from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import validate_model, load_model
from src.data_loader import get_loaders
from src.logger import Logger


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