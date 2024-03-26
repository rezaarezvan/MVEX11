import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Pretrained_ViT(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(Pretrained_ViT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        for params in self.vit.parameters():
            params.requires_grad = False

        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        self.vit(x)
        return x

    def _forward(self, x):
        attention_weights = self.vit._process_input(x)
        n = attention_weights.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        attention_weights = torch.cat(
            (batch_class_token, attention_weights), dim=1)

        for i in range(12):
            attention_weights = self.vit.encoder.layers[i](attention_weights)

        return attention_weights
