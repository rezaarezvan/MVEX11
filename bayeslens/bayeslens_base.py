import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from data import load_data
from utils.entrop import test_model_noise, calculate_weighted_averages, plot_weighted_averages, plot_entropy_prob, compute_k
from utils.model import load_model, save_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(0, 0.1))
        self.weight_sigma = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(0, 0.1))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_sigma = nn.Parameter(
            torch.Tensor(out_features).normal_(0, 0.1))

    def forward(self, x):
        weight = Normal(self.weight_mu, F.softplus(
            self.weight_sigma)).rsample()
        bias = Normal(self.bias_mu, F.softplus(self.bias_sigma)).rsample()
        return F.linear(x, weight, bias)


class BayesLensCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(BayesLensCNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc1 = BayesianLinear(256*256, 1024)
        self.fc2 = BayesianLinear(1024, 512)
        self.fc3 = BayesianLinear(512, 128)
        self.fc4 = BayesianLinear(128, num_classes)
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


def train_model(model, train, test, optimizer, criterion, num_epochs=40):
    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train)}], Loss: {loss.item():.4f}')

        evaluate_model(model, test)
        print()


def evaluate_model(model, test, mode='validation'):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'{mode.capitalize()} accuracy of the model: {accuracy} %')


def main():
    dataset_path = '../extra/datasets/SODA'
    train, val, test = load_data(dataset_path, batch_size=16)

    model = BayesLensCNN(num_classes=6)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train, val, optimizer, criterion)

    iterations = 2
    entropies = []
    weighted_average = []

    sigmas = np.arange(0.1, 1, 0.1)
    sigmas = np.append(sigmas, [1, 10, 100, 500])

    for sigma in sigmas:
        print(f"Sigma: {sigma}")
        entropy = test_model_noise(model, test, sigma=sigma, iters=iterations)
        weighted_average.append((calculate_weighted_averages(entropy), sigma))
        entropies.append(entropy)
        for lam in [0.1, 0.5, 1]:
            print(f"Lambda: {lam}, K: {compute_k(entropy, _lambda=lam)}")
        print('-----------------------------------\n')

    plot_weighted_averages(weighted_average)
    for entropy, sigma in zip(entropies, sigmas):
        plot_entropy_prob(entropy, sigma, 10,
                          iterations)


if __name__ == "__main__":
    main()
