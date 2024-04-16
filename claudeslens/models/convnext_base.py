import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNext(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(ConvNext, self).__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

        last_block = list(self.convnext.features)[-1]
        self.lastconv_output_channels = [layer for layer in last_block.modules(
        ) if isinstance(layer, nn.Conv2d)][-1].out_channels

        for param in self.convnext.parameters():
            param.requires_grad = False

        self.convnext.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.lastconv_output_channels, num_classes)
        )

        self.saved_feature_maps = {}

    def forward(self, x):
        handle = self.convnext.features[-1].register_forward_hook(
            self.save_feature_maps_hook)
        result = self.convnext(x)
        handle.remove()

        return result

    def save_feature_maps_hook(self, module, input, output):
        if 'last_conv_output' not in self.saved_feature_maps:
            self.saved_feature_maps['last_conv_output'] = []
        self.saved_feature_maps['last_conv_output'].append(output.detach())
