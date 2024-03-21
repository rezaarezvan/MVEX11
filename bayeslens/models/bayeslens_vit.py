import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class BayesLens_ViT(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(BayesLens_ViT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.classifier = nn.Linear(
            self.vit.heads.head.in_features, num_classes)
        self.vit.heads.head = self.classifier
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.vit(x)
        x = self.softmax(x)
        return x
