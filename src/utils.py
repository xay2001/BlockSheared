import torch  # 导入 PyTorch 库，用于张量计算和深度学习模型的实现
from src.data_loader import get_loaders
from transformers import AutoTokenizer  # 使用AutoTokenizer加载tokenizer
import torch.nn.functional as F
def accuracy(predictions, labels):  # 定义 accuracy 函数，用于计算模型的预测准确率
    _, preds = torch.max(predictions, dim=1)  # 找到预测结果中的最大值对应的类别索引
    return (preds == labels).float().mean()  # 计算预测结果与实际标签匹配的比例，即准确率

def save_model(model, path):  # 定义 save_model 函数，用于将训练好的模型保存到指定路径
    torch.save(model.state_dict(), path)  # 使用 torch.save 将模型的状态字典保存到指定路径

def load_model(model, path):  # 定义 load_model 函数，用于从指定路径加载模型的状态字典
    model.load_state_dict(torch.load(path))  # 使用 torch.load 加载模型的状态字典到当前模型
    return model  # 返回加载了状态字典的模型



def validate_model(model, val_data, logger):
    model.eval()  # 设置模型为评估模式，以禁用 dropout 等训练时特有的操作
    total_loss = 0  # 初始化总损失
    total_correct = 0  # 初始化正确预测的 token 数
    total_tokens = 0  # 初始化总 token 数

    with torch.no_grad():  # 禁用梯度计算以加快推理速度并减少内存消耗
        inputs = val_data.input_ids.to(model.device)  # 将验证集输入移至模型的设备（例如 GPU）

        # 逐批次进行前向传播，假设 val_data 是一个很大的序列，我们可以选择分块处理
        for i in range(0, inputs.size(1), model.config.n_positions):
            # 获取当前块的输入
            input_ids = inputs[:, i:i + model.config.n_positions]

            # 对当前块进行前向传播，获取模型输出
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss  # Hugging Face 模型的输出包含 loss
            logits = outputs.logits  # 模型的 logits 输出

            # 计算 token 数量
            num_tokens = input_ids.numel()
            total_loss += loss.item() * num_tokens  # 按 token 数量累计损失
            total_tokens += num_tokens

            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)  # 获取预测的 token
            correct_predictions = (predictions == input_ids).float()  # 计算正确预测的 token 数
            total_correct += correct_predictions.sum().item()  # 累计正确预测的 token 数

    avg_loss = total_loss / total_tokens  # 计算平均损失
    accuracy = total_correct / total_tokens  # 计算整体准确率
    perplexity = torch.exp(torch.tensor(avg_loss))  # 计算困惑度（Perplexity）

    # 将结果记录到日志
    logger.log(f"Validation Loss: {avg_loss:.4f}")
    logger.log(f"Validation Perplexity: {perplexity:.4f}")
    logger.log(f"Validation Accuracy: {accuracy:.4f}")

    return avg_loss, perplexity, accuracy


def log_training_time(epoch, time_taken, logger):  # 定义 log_training_time 函数，用于记录每个 epoch 的训练时间
    logger.log(f"Epoch {epoch} took {time_taken:.2f} seconds")  # 记录并输出当前 epoch 所用的时间（以秒为单位）

def log_memory_usage(logger):  # 定义 log_memory_usage 函数，用于记录 GPU 内存的使用情况
    mem_usage = torch.cuda.memory_allocated() / (1024 ** 3)  # 计算当前 GPU 已分配内存，单位为 GB
    logger.log(f"GPU Memory Usage: {mem_usage:.2f} GB")  # 记录并输出 GPU 内存的使用情况
