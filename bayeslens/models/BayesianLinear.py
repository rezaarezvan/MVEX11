import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, mu=0, sigma=0.1):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(mu, sigma))
        self.weight_sigma = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(mu, sigma))

        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).normal_(mu, sigma))
        self.bias_sigma = nn.Parameter(
            torch.Tensor(out_features).normal_(mu, sigma))

    def forward(self, x):
        weight = Normal(self.weight_mu, F.softplus(
            self.weight_sigma)).rsample()
        bias = Normal(self.bias_mu, F.softplus(self.bias_sigma)).rsample()
        return F.linear(x, weight, bias)
