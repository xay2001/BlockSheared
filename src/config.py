import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:  # 指定文件编码为 utf-8
        config = yaml.safe_load(f)
    return config