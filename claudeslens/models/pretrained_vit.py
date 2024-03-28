from .vit_b_16 import Pretrained_ViT


class Pretrained_ViT_B_16(Pretrained_ViT):
    def __init__(self, num_classes=6, num_channels=None, num_inputs=None):
        super(Pretrained_ViT_B_16, self).__init__()
