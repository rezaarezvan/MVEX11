import torch.nn as nn
from .BayesianLinear import BayesianLinear


class BayesLens(nn.Module):
    def __init__(self, num_inputs, num_classes, hidden_size1=1024, hidden_size2=512, hidden_size3=128, dropout_rate=0.1):
        super(BayesLens, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = BayesianLinear(num_inputs, hidden_size1)
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
