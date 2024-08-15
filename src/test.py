from src.utils import load_model, validate_model
from src.logger import Logger
from transformers import AutoModelForCausalLM, AutoTokenizer  # 使用AutoModel和AutoTokenizer加载模型


def evaluate_model(config):
    logger = Logger(config['log_path'])

    if config['use_local_model']:
        # 使用本地路径加载 LLaMA 2-7B 模型和 tokenizer
        model_path = config['local_model_path']
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        # 使用在线路径加载 LLaMA 2-7B 模型和 tokenizer
        model_name = config['hf_model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model = load_model(model, config['save_path'])

    validate_model(model, config['dataset_name'], 'validation', logger)
    logger.log("Evaluation complete.")