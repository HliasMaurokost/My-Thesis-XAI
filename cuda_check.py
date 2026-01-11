# Έλεγχος διαθεσιμότητας και πληροφοριών CUDA μέσω PyTorch
import torch

print('CUDA available:', torch.cuda.is_available())
print('CUDA runtime version (από PyTorch):', torch.version.cuda)
print('Driver capability (GPU compute capability):',
      torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A')