import time
from src.blockllm import BlockLLM
from src.data_loader import load_data
from src.utils import log_training_time, log_memory_usage, save_model
from src.logger import Logger


def fine_tune_model(config):
    logger = Logger(config['log_path'])
    model = ...  # 加载模型，例如 BertModel.from_pretrained('bert-base-uncased')
    train_loader = load_data(config['dataset_name'], 'train', config['batch_size'])

    block_llm = BlockLLM(model, config['sparsity'], config['patience'], config['learning_rate'], logger)

    for epoch in range(config['epochs']):
        start_time = time.time()
        total_loss = 0
        for batch in train_loader:
            loss = block_llm.train_step(batch)
            total_loss += loss

        epoch_time = time.time() - start_time
        log_training_time(epoch, epoch_time, logger)
        log_memory_usage(logger)

        logger.log(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

    save_model(model, config['save_path'])
    logger.log(f"Model saved to {config['save_path']}")
    return model
