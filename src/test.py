from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import validate_model, load_model
from src.data_loader import get_loaders
from src.logger import Logger


def evaluate_model(config):
    logger = Logger(config['log_path'])

    # 加载模型和 tokenizer 的代码

    model = load_model(model, config['save_path'])

    _, val_enc = get_loaders(
        config['dataset_name'],
        nsamples=config['batch_size'],
        seed=42,
        seqlen=2048,
        tokenizer=tokenizer,
        use_local=config['use_local_data'],
        local_paths=config['local_data_paths']
    )

    avg_loss, perplexity, accuracy = validate_model(model, val_enc, logger)
    logger.log(f"Evaluation complete. Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.4f}")
