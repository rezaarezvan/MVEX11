import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, num_inputs, num_classes, num_channels):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_inputs, num_classes, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x
