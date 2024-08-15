from src.utils import load_model, validate_model
from src.logger import Logger

def evaluate_model(config):
    logger = Logger(config['log_path'])
    model = ...  # 初始化模型，例如 BertModel.from_pretrained('bert-base-uncased')
    model = load_model(model, config['save_path'])
    logger.log(f"Model loaded from {config['save_path']}")

    validate_model(model, config['dataset_name'], 'validation', logger)
    logger.log("Evaluation complete.")
