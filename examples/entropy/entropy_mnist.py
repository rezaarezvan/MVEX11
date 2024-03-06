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


def save_model_parameters(model):
    original_params = [param.clone() for param in model.parameters()]
    return original_params


@torch.no_grad()
def restore_model_parameters(model, original_params):
    for param, original in zip(model.parameters(), original_params):
        param.copy_(original)


@torch.no_grad()
def add_noise(model, sigma):
    for param in model.parameters():
        param += torch.randn_like(param) * sigma
    return model


def calculate_entropy(predictions: torch.Tensor):
    """
    Follows this entropy function:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    log_e
    The function works by first calculating the quantity of the number
    in the list and creates a list where it's frequency is placed in that list.
    For example:
    [1,7,1,7,1]->[0,3,0,0,0,0,0,2]/len([1,7,1,7,1])->[0,(3/5),0,0,0,0,0,(2/5)]
    -------------------------------------------------------------------------
    Perform entropy calculation on every row in the predictions matrix
    """
    entropy = torch.zeros(predictions.size(0))
    for i in range(predictions.size(0)):
        probability  = torch.bincount(predictions[i], minlength=10).float()
        probability /= predictions.size(1) # Normalize
        entropy[i]   = Categorical(probs=probability).entropy()
    return entropy

@torch.no_grad()
def test_model_noise(model, dataset, sigma=0, iters=10):
    model.eval()
    result = []

    print(f"Testing model with noise scale {sigma:.2f}...")
    for batch, (data, target) in enumerate(dataset):
        predictions = []
        data, target = data.to(DEVICE), target.to(DEVICE)
        for _ in range(iters):
            original_params = save_model_parameters(model)
            add_noise(model, sigma)

            output = model(data.view(-1, 28 * 28))
            pred   = torch.softmax(output, dim=1).argmax(dim=1)

            predictions.append(pred)
            restore_model_parameters(model, original_params)
        
        pred_matrix = torch.stack(predictions, dim=1)
        entropy     = calculate_entropy(pred_matrix)
        accuracy    = (pred_matrix == target.view(-1, 1)).float().mean(dim=1)
        
        for e, a in zip(entropy, accuracy):
            result.append((e.item(), a.item()))

        if (batch + 1) % 50 == 0:
            print(f'Batch {batch + 1}/{len(dataset)} completed')

    print(f'Batch {batch+1}/{len(dataset)} completed')
    print(f'Testing model with noise scale {sigma:.2f} completed\n')
    return result


def plot_entropy_prob(ent_prob, sigma, acc, iterations):
    """
    Plots the list of probability/entropy parts on the x/y-axis respectively.
    """
    entropy, probability = zip(*ent_prob)

    plt.figure(figsize=(10, 6))
    plt.scatter(probability, entropy, alpha=0.1)
    plt.xlabel('Probability')
    plt.ylabel('Entropy')
    plt.title(f'Entropy to Probability (sigma={sigma}, iterations={iterations}, {acc}%)')
    plt.grid()
    plt.ylim(-0.1, np.log(10) + 0.1)
    plt.xlim(-0.1, 1.1)
    plot_curve(classes=10)
    if SAVE_PLOT: plt.savefig(f'plots/entropy_prob_sigma_{sigma:.2f}.pdf')
    else: plt.show()


def plot_curve(classes):
    """
    This call plots the domain of the amount of given classes.
    (Beware there exists off-by-one, this is handled in the function)
    """
    classes -= 1
    # Arc-function at bottom
    x = np.linspace(0.0001, 0.9999, 10000)
    y = -x * np.log(x) - (1 - x) * np.log(1 - x)
    plt.plot(x, y, color='orange')
    x = np.linspace(0.0001, 0.9999, 10000)
    y = -x * np.log(x) - (1 - x) * np.log(1 - x)

    # Higher top function
    y2 = []
    for p in x:
        a = -p * np.log(p)
        for i in range(classes):
            a -= (1 - p) / (classes) * np.log((1 - p) / classes)
        y2.append(a)

    plt.plot(x, y, color="orange")
    plt.plot(x, y2, color="orange")

    # This plots the vertical line
    plt.plot([0, 0], [0, y2[0]], color="orange")


def save_model(model, path=model_path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path=model_path):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model

def main():
    n_inputs = 28 * 28
    n_outputs = 10
    
    train, test = load_data()

    model = LogisticRegression(n_inputs, n_outputs)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    if LOAD_WEIGHTS:
        model = load_model(model)
    else:
        train_model(model, train, test, optimizer, criterion, epochs=1)
    
    if SAVE_WEIGHTS:
        save_model(model)
    
    accuracy = test_model(model, test)
    iterations = 100
    entropies = []

    sigmas = np.arange(0, 1, 0.25)
    sigmas = np.append(sigmas, [1, 10, 100, 500])
    for sigma in sigmas:
        entropy = test_model_noise(model, test, sigma=sigma, iters=iterations)
        entropies.append(entropy)
    
    for entropy, sigma in zip(entropies, sigmas):
        plot_entropy_prob(entropy, sigma, accuracy, iterations)


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
