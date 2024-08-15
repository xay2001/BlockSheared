import argparse
from src.config import load_config
from src.train import fine_tune_model
from src.test import evaluate_model
from src.wandb_integration import init_wandb

def main():
    parser = argparse.ArgumentParser(description='BlockLLM Training and Evaluation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Run mode: train or test')
    args = parser.parse_args()

    config = load_config(args.config)
    init_wandb(config)  # 初始化W&B

    if args.mode == 'train':
        fine_tune_model(config)
    elif args.mode == 'test':
        evaluate_model(config)

if __name__ == '__main__':
    main()