import torch

print("CUDA mevcut mu:", torch.cuda.is_available())
print("GPU sayısı:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU ismi:", torch.cuda.get_device_name(0))