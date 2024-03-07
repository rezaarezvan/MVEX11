import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import sys
from utils.utils import load_model, save_model, test_model_noise, calculate_weighted_averages, plot_weighted_averages, plot_entropy_prob

model_path = 'model_state/entropy_mnist.pth'

SAVE_PLOT    = True if '-s'  in sys.argv else False
LOAD_WEIGHTS = True if '-lw' in sys.argv else False
SAVE_WEIGHTS = True if '-sw' in sys.argv else False


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)

def load_data():
    train = torchvision.datasets.MNIST(
        root='../../extra/datasets', train=True, download=True, transform=transforms.ToTensor())
    test = torchvision.datasets.MNIST(
        root='../../extra/datasets', train=False, download=True, transform=transforms.ToTensor())

    train = DataLoader(train, batch_size=32, shuffle=True)
    test = DataLoader(test, batch_size=32, shuffle=False)

    return train, test

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


def train_model(model, train, test, optimizer, criterion, epochs=3):

    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images.flatten(1))
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train)}], Loss: {loss.item():.4f}')

        print(f'Epoch {epoch + 1} completed')
        print('Testing model...')
        test_model(model, test)

@torch.no_grad()
def test_model(model, test):
    model.eval()
    correct = 0

    for images, labels in test:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images.flatten(start_dim=1))
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()

    acc = 100. * correct / len(test.dataset)
    print(f'Accuracy: {correct}/{len(test.dataset)} ({acc: .2f} %)\n')
    return acc

def main():
    n_inputs = 28 * 28
    n_outputs = 10
    
    train, test = load_data()

    model = LogisticRegression(n_inputs, n_outputs)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    if LOAD_WEIGHTS:
        model = load_model(model, model_path)
    else:
        train_model(model, train, test, optimizer, criterion, epochs=1)
    
    if SAVE_WEIGHTS:
        save_model(model, model_path)
    
    accuracy = test_model(model, test)
    iterations = 100
    entropies = []

    weighted_average = []
    sigmas = np.arange(0, 1, 0.1)
    sigmas = np.append(sigmas, [1, 10, 100, 500])
    for sigma in sigmas:
        entropy = test_model_noise(model, test, sigma=sigma, iters=iterations)
        weighted_average.append((calculate_weighted_averages(entropy), sigma))
        entropies.append(entropy)
   
    plot_weighted_averages(weighted_average)
    for entropy, sigma in zip(entropies, sigmas):
        plot_entropy_prob(entropy, sigma, accuracy, iterations, SAVE_PLOT=SAVE_PLOT)


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        print('Usage: python entropy_mnist.py [-s] [-lw] [-sw] [-h]')
        print('Options:')
        print('  -s:  Save plots')
        print('  -lw: Load weights')
        print('  -sw: Save weights')
        print('  -h:  Help')
        sys.exit(0)
    main()
