import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ClaudesLensCNN(nn.Module):
    def __init__(self, num_channels=3, num_inputs=256*256*3, num_classes=6, dropout_rate=0.2):
        super(ClaudesLensCNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_extractor = nn.Sequential(
            ConvBlock(num_channels, 16),
            ConvBlock(16, 32),
        )
        self.flatten = nn.Flatten()
        img_size = int((num_inputs/num_channels)**0.5)
        with torch.no_grad():
            self._dummy_input = torch.zeros(
                1, num_channels, img_size, img_size)
            self._flattened_size = self._forward_features(
                self._dummy_input).shape[1]

        self.classifier = nn.Linear(self._flattened_size, num_classes)

    def _forward_features(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
