import torch.nn as nn
from .BayesianLinear import BayesianLinear


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


class BayesLensCNN(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(BayesLensCNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
        )
        self.flatten = nn.Flatten()
        self.classifier = BayesianLinear(32*64*64, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
