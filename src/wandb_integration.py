import wandb

def init_wandb(config):
    wandb.init(project=config['project_name'], config=config)

def log_metrics(metrics):
    wandb.log(metrics)
