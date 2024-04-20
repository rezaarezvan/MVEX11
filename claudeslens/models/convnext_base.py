import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNext(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(ConvNext, self).__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.feature_maps = {}

        for param in self.convnext.parameters():
            param.requires_grad = False

        last_block = list(self.convnext.features)[-1]
        self.lastconv_output_channels = [layer for layer in last_block.modules(
        ) if isinstance(layer, nn.Conv2d)][-1].out_channels
        self.convnext.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.lastconv_output_channels, num_classes)
        )

    def _initialize_hooks(self):
        for i, layer in enumerate(self.convnext.features):
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Conv2d):
                        sublayer.register_forward_hook(
                            self.save_feature_map_hook(i))

    def save_feature_map_hook(self, layer_idx):
        def hook(module, input, output):
            self.feature_maps[layer_idx] = output.detach()
        return hook

    def forward(self, x, return_features=False):
        if return_features:
            self._initialize_hooks()
        return self.convnext(x)
