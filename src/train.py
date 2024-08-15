import time
from src.blockllm import BlockLLM
from src.data_loader import get_loaders
from src.utils import log_training_time, log_memory_usage, save_model
from src.logger import Logger
from transformers import AutoModelForCausalLM, AutoTokenizer  # 使用AutoModel和AutoTokenizer加载模型


def fine_tune_model(config):
    logger = Logger(config['log_path'])

    if config['use_local_model']:
        model_path = config['local_model_path']
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model_name = config['hf_model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

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

    block_llm = BlockLLM(model, config['sparsity'], config['patience'], config['learning_rate'], logger)

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

    save_model(model, config['save_path'])
    logger.log(f"Model saved to {config['save_path']}")
    return model