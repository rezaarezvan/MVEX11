import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class CustomEncoderBlock(nn.Module):
    def __init__(self, original_encoder_block):
        super().__init__()
        self.encoder_block = original_encoder_block

    def forward(self, input, need_weights=False):
        x = self.encoder_block.ln_1(input)
        if need_weights:
            x, attn_weights = self.encoder_block.self_attention(
                x, x, x, need_weights=True)
        else:
            x, _ = self.encoder_block.self_attention(
                x, x, x, need_weights=False)
        x = self.encoder_block.dropout(x)
        x = x + input

        y = self.encoder_block.ln_2(x)
        y = self.encoder_block.mlp(y)

        if need_weights:
            return x + y, attn_weights

        return x + y, None


class CustomEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.pos_embedding = original_encoder.pos_embedding
        self.dropout = original_encoder.dropout
        self.layers = nn.ModuleList(
            [CustomEncoderBlock(layer) for layer in original_encoder.layers]
        )
        self.ln = original_encoder.ln

    def forward(self, x, need_weights=False):
        attn_maps = []
        x = x + self.pos_embedding
        x = self.dropout(x)

        for layer in self.layers:
            x, attn_map = layer(x, need_weights=need_weights)
            if attn_map is not None:
                attn_maps.append(attn_map)

        if need_weights:
            attn_maps = torch.stack(attn_maps, dim=1)
            return self.ln(x), attn_maps

        return self.ln(x), None


class Pretrained_ViT(nn.Module):
    def __init__(self, num_classes=6, num_inputs=None, num_channels=None):
        super(Pretrained_ViT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        original_encoder = self.vit.encoder
        self.vit.encoder = CustomEncoder(original_encoder)

        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.heads.head = nn.Linear(
            self.vit.heads.head.in_features, num_classes)

    def forward(self, x, need_weights=False):
        x = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x, attn_maps = self.vit.encoder(x, need_weights=need_weights)

        x = x[:, 0]
        x = self.vit.heads(x)

        if need_weights:
            return x, attn_maps

        return x
