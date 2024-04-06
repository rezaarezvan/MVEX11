import torch

SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
