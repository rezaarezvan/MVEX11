import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Pretrained_ViT(nn.Module):
    def __init__(self):
        super(Pretrained_ViT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    def forward(self, x):
        x = self.vit(x)
        return x
