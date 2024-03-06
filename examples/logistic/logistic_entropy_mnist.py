import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

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
    model.to(DEVICE)

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
        acc = test_model(model, test)
        print(acc)


def test_model(model, test):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images.flatten(start_dim=1))
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    acc = 100. * correct / len(test.dataset)
    print(f'Accuracy: {correct}/{len(test.dataset)} ({acc: .2f} %)')
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


def calculate_entropy(predictions):
    """
    Follows this entropy function:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    log_e
    The function works by first calculating the quantity of the number
    in the list and creates a list where it's frequency is placed in that list.
    For example:
    [1,7,1,7,1]->[0,3,0,0,0,0,0,2]/len([1,7,1,7,1])->[0,(3/5),0,0,0,0,0,(2/5)]
    """
    value_counts = torch.bincount(predictions) / len(predictions)
    entropy = -np.sum([p * np.log(p) for p in value_counts if p > 0])
    return entropy


@torch.no_grad()
def test_model_noise(model, test, sigma, iterations):
    model.eval()
    entropy_probability = []

    print(f'Testing model with sigma = {sigma}, iterations = {iterations}')

    for batch, (images, labels) in enumerate(test):
        for image, label in zip(images, labels):
            image, label = image.to(DEVICE), label.to(DEVICE)

            predictions = torch.zeros(iterations, dtype=torch.int32)

            for i in range(iterations):
                original_weights = save_model_parameters(model)
                add_noise(model, sigma)

                outputs = model(image.view(-1, 28 * 28))
                pred = outputs.argmax(dim=1, keepdim=True)
                predictions[i] = pred.item()

                restore_model_parameters(model, original_weights)

            entropy = calculate_entropy(predictions)
            probability = torch.bincount(predictions, minlength=10)[
                              label.item()] / len(predictions)
            entropy_probability.append((entropy, probability))

        if (batch + 1) % 50 == 0:
            print(f'Batch {batch + 1}/{len(test)} completed')

    return entropy_probability


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
    plt.ylim(0, np.log10(10) + 0.1)
    plt.xlim(-0.1, 1.1)
    plot_curve(classes=10)
    plt.show()


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
    x = np.linspace(0.001, 0.999, 1000)
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

def main():
    print(calculate_entropy(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    n_inputs = 28 * 28
    n_outputs = 10
    train, test = load_data()
    model = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train, test, optimizer, criterion, epochs=1)
    accuracy = test_model(model, test)
    iterations = 100
    #sigmas = [x+2 for x in range(5)]
    #for sigma in sigmas:
    plot_entropy_prob(test_model_noise(model, test, sigma=0, iterations=iterations), 0, accuracy,iterations)


if __name__ == '__main__':
    main()
