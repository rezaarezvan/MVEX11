import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNext(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(ConvNext, self).__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

        last_block = list(self.convnext.features.children())[-4]
        if isinstance(last_block, nn.Sequential):
            last_conv_layer = [layer for layer in last_block.modules(
            ) if isinstance(layer, nn.Conv2d)][-1]
            self.lastconv_output_channels = last_conv_layer.out_channels
        else:
            print(
                "Check the model structure, the assumption about the last block might be incorrect.")
            lastconv_output_channels = 512
            self.lastconv_output_channels = lastconv_output_channels

        self.convnext.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.lastconv_output_channels, num_classes)
        )

    def forward(self, x):
        return self.convnext(x)
