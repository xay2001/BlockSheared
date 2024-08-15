import time
from src.blockllm import BlockLLM
from src.data_loader import load_data
from src.utils import log_training_time, log_memory_usage, save_model
from src.logger import Logger
from transformers import AutoModelForCausalLM, AutoTokenizer  # 使用AutoModel和AutoTokenizer加载模型


def fine_tune_model(config):
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

    train_loader = load_data(config['dataset_name'], 'train', config['batch_size'])

    block_llm = BlockLLM(model, config['sparsity'], config['patience'], config['learning_rate'], logger)

    for epoch in range(config['epochs']):
        start_time = time.time()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input_ids'].to(model.device)
            loss = block_llm.train_step({'input': inputs, 'label': inputs})
            total_loss += loss

        epoch_time = time.time() - start_time
        log_training_time(epoch, epoch_time, logger)
        log_memory_usage(logger)

        logger.log(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

    save_model(model, config['save_path'])
    logger.log(f"Model saved to {config['save_path']}")
    return model