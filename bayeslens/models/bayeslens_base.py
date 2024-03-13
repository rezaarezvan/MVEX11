import torch.nn as nn
from .BayesianLinear import BayesianLinear


class BayesLens(nn.Module):
    def __init__(self, input_size=256*256*3, num_classes=6, hidden_size1=1024, hidden_size2=512, hidden_size3=128, dropout_rate=0.5):
        super(BayesLens, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = BayesianLinear(input_size, hidden_size1)
        self.fc2 = BayesianLinear(hidden_size1, hidden_size2)
        self.fc3 = BayesianLinear(hidden_size2, hidden_size3)
        self.fc4 = BayesianLinear(hidden_size3, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
