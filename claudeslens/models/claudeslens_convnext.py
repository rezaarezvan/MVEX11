import torch.nn as nn
from .convnext_base import ConvNext


class ClaudesLens_ConvNext(ConvNext):
    def __init__(self, num_classes=6, num_channels=None, num_inputs=None, hidden_size_1=1024, hidden_size_2=512):
        super(ClaudesLens_ConvNext, self).__init__(
            num_classes, num_channels, num_inputs)

        self.convnext.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.lastconv_output_channels, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, num_classes)
        )
