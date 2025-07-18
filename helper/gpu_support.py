import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Prints GPU name (e.g., NVIDIA GeForce RTX 3060)
print(torch.version.cuda)  # Prints PyTorch’s CUDA version