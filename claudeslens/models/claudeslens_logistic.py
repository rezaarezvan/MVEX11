import torch.nn as nn


class ClaudesLens_Logistic(nn.Module):
    def __init__(self, num_inputs, num_classes, num_channels):
        super(ClaudesLens_Logistic, self).__init__()
        self.linear = nn.Linear(num_inputs, num_classes, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x
