import torch

def monitor_memory_usage():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    memory_usage = allocated_memory / total_memory * 100
    print(f"GPU Memory Usage: {memory_usage:.2f}%")
