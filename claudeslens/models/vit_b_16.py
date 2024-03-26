import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16, ViT_B_16_Weights


class CustomEncoderBlock(nn.Module):
    def __init__(self, original_encoder_block):
        super().__init__()
        self.encoder_block = original_encoder_block

    def forward(self, input):
        x = self.encoder_block.ln_1(input)
        x, attn_weights = self.encoder_block.self_attention(
            x, x, x, need_weights=True)
        x = self.encoder_block.dropout(x)
        x = x + input

        y = self.encoder_block.ln_2(x)
        y = self.encoder_block.mlp(y)
        return x + y, attn_weights


class CustomEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.pos_embedding = original_encoder.pos_embedding
        self.dropout = original_encoder.dropout
        self.layers = nn.ModuleList(
            [CustomEncoderBlock(layer) for layer in original_encoder.layers]
        )
        self.ln = original_encoder.ln

    def forward(self, x):
        attn_maps = []
        x = x + self.pos_embedding
        x = self.dropout(x)
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_maps.append(attn_map)
        return self.ln(x), attn_maps


class Pretrained_ViT(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(Pretrained_ViT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # Replace the encoder with a custom one
        original_encoder = self.vit.encoder
        self.vit.encoder = CustomEncoder(original_encoder)

        # Freeze the parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace the classification head
        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, num_classes)

    def forward(self, x, need_weights=False):
        x = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x, attn_maps = self.vit.encoder(x)

        x = x[:, 0]
        x = self.vit.heads(x)
        if need_weights:
            attn_maps = torch.stack(attn_maps, dim=1)
            return x, attn_maps
        return x

    def visualize_attention_map(self, image, attention_map):
        attention_map = attention_map.mean(dim=1)
        attention_map_no_class_token = attention_map[:, 1:, 1:]
        attention_map_single = attention_map_no_class_token[0]
        attention_map_avg = attention_map_single.mean(dim=0)

        seq_len = attention_map_avg.shape[0]
        sqrt_len = int(np.sqrt(seq_len))
        attention_map_2d = attention_map_avg.view(sqrt_len, sqrt_len)

        attention_map_resized = torch.nn.functional.interpolate(
            attention_map_2d.unsqueeze(0).unsqueeze(0),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        attention_map_np = attention_map_resized.detach().cpu().numpy()

        plt.imshow(attention_map_np, cmap='jet',
                   alpha=0.5, interpolation='nearest')
        plt.colorbar()
        plt.show()
