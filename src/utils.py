import torch
from src.data_loader import load_data

def accuracy(predictions, labels):
    _, preds = torch.max(predictions, dim=1)
    return (preds == labels).float().mean()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def validate_model(model, dataset_name, split, logger):
    val_loader = load_data(dataset_name, split, batch_size=32)
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['input'])
            acc = accuracy(outputs, batch['label'])
            total_accuracy += acc.item()
    logger.log(f"Validation Accuracy: {total_accuracy / len(val_loader)}")
