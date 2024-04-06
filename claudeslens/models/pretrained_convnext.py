from .convnext_base import ConvNext


class Pretrained_ConvNext(ConvNext):
    def __init__(self, num_classes=6, num_channels=None, num_inputs=None):
        super(Pretrained_ConvNext, self).__init__(
            num_classes, num_channels, num_inputs)
