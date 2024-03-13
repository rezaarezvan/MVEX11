import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from .BayesianLinear import BayesianLinear


class BayesLens_ViT(nn.Module):
    def __init__(self, num_classes=6):
        super(BayesLens_ViT, self).__init__()
        self.classifier = BayesianLinear(768, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = self.classifier

    def forward(self, x):
        x = self.vit(x)
        x = self.softmax(x)
        return x
