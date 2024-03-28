import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNext(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(ConvNext, self).__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

        # TODO: Fix this, doesn't work for now
        self.convnext.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.convnext.lastconv_output_channels, num_classes)
        )

    def forward(self, x):
        return self.convnext(x)
