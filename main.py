import argparse  # 导入argparse模块，用于处理命令行参数
from src.config import load_config  # 从src.config模块中导入load_config函数，用于加载配置文件
from src.train import fine_tune_model  # 从src.train模块中导入fine_tune_model函数，用于微调模型
from src.test import evaluate_model  # 从src.test模块中导入evaluate_model函数，用于测试模型性能
from src.wandb_integration import init_wandb  # 从src.wandb_integration模块中导入init_wandb函数，用于初始化W&B

def main():  # 定义主函数
    parser = argparse.ArgumentParser(description='BlockLLM Training and Evaluation')  # 创建ArgumentParser对象，用于解析命令行参数
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')  # 添加命令行参数--config，指定配置文件路径
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Run mode: train or test')  # 添加命令行参数--mode，用于选择运行模式：train（训练）或test（测试）
    args = parser.parse_args()  # 解析命令行参数并存储在args变量中

    config = load_config(args.config)  # 加载配置文件，返回配置字典
    init_wandb(config)  # 初始化W&B（Weights & Biases）实验追踪

    if args.mode == 'train':  # 如果选择了train模式
        fine_tune_model(config)  # 调用fine_tune_model函数进行模型微调
    elif args.mode == 'test':  # 如果选择了test模式
        evaluate_model(config)  # 调用evaluate_model函数进行模型性能测试

if __name__ == '__main__':  # 如果当前脚本是作为主程序运行
    main()  # 调用主函数
