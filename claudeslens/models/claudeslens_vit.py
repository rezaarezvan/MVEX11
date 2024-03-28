import torch.nn as nn
from .vit_b_16 import Pretrained_ViT


class ClaudesLens_ViT(Pretrained_ViT):
    def __init__(self, num_classes=6, num_channels=None, num_inputs=None, hidden_size_1=1024, hidden_size_2=512):
        super(ClaudesLens_ViT, self).__init__(
            num_classes, num_channels, num_inputs)

        self.classifier = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, num_classes)
        )

        self.vit.heads.head = self.classifier
