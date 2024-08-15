import torch  # 导入 PyTorch 库，用于张量计算和深度学习模型的实现
from src.data_loader import load_data  # 从 src.data_loader 模块中导入 load_data 函数，用于加载数据集
from transformers import AutoTokenizer  # 使用AutoTokenizer加载tokenizer
def accuracy(predictions, labels):  # 定义 accuracy 函数，用于计算模型的预测准确率
    _, preds = torch.max(predictions, dim=1)  # 找到预测结果中的最大值对应的类别索引
    return (preds == labels).float().mean()  # 计算预测结果与实际标签匹配的比例，即准确率

def save_model(model, path):  # 定义 save_model 函数，用于将训练好的模型保存到指定路径
    torch.save(model.state_dict(), path)  # 使用 torch.save 将模型的状态字典保存到指定路径

def load_model(model, path):  # 定义 load_model 函数，用于从指定路径加载模型的状态字典
    model.load_state_dict(torch.load(path))  # 使用 torch.load 加载模型的状态字典到当前模型
    return model  # 返回加载了状态字典的模型


def validate_model(model, dataset_name, split, logger):
    if config['use_local_model']:
        model_path = config['local_model_path']
    else:
        model_path = config['hf_model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    val_loader = load_data(dataset_name, split, batch_size=32)
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = tokenizer(batch['input'], return_tensors='pt', padding=True, truncation=True).input_ids
            inputs = inputs.to(model.device)
            outputs = model(inputs)
            acc = accuracy(outputs.logits, batch['label'])
            total_accuracy += acc.item()
    logger.log(f"Validation Accuracy: {total_accuracy / len(val_loader)}")

def log_training_time(epoch, time_taken, logger):  # 定义 log_training_time 函数，用于记录每个 epoch 的训练时间
    logger.log(f"Epoch {epoch} took {time_taken:.2f} seconds")  # 记录并输出当前 epoch 所用的时间（以秒为单位）

def log_memory_usage(logger):  # 定义 log_memory_usage 函数，用于记录 GPU 内存的使用情况
    mem_usage = torch.cuda.memory_allocated() / (1024 ** 3)  # 计算当前 GPU 已分配内存，单位为 GB
    logger.log(f"GPU Memory Usage: {mem_usage:.2f} GB")  # 记录并输出 GPU 内存的使用情况
