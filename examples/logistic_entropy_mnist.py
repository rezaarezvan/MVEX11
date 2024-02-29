import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

torch.manual_seed(0)


def load_data():
    trainset = torchvision.datasets.MNIST(
        root='../extra/datasets', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(
        root='../extra/datasets', train=False, download=True, transform=transforms.ToTensor())

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader


class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


def train_model(model, train, test, optimizer, epochs=3):
    loss = nn.CrossEntropyLoss()
    acc = 0

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train):
            optimizer.zero_grad()
            input = model(data.view(-1, 28*28))
            output = loss(input, target)
            output.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {output.item()}')

        print(f'Epoch {epoch+1} completed')
        # print('Testing model...')
        # acc = test_model(model, test)
        # print('')

    return acc


def test_model(model, test):
    loss = nn.CrossEntropyLoss()

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test:
            input = model(data.view(-1, 28*28))
            pred = input.argmax(dim=1, keepdim=True)

            test_loss += loss(input, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test.dataset)
    acc = 100. * correct / len(test.dataset)
    print(
        f'Average loss: {test_loss: .4f}, Accuracy: {correct}/{len(test.dataset)} ({acc: .2f} %)')

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
    return Categorical(probs=torch.bincount(predictions)).entropy()


@torch.no_grad()
def test_model_noise(model, test, sigma, iterations=10, num_pics=100):
    model.eval()
    ent_prob = []

    for data, target in test:
        for x, y in zip(data, target):
            if len(ent_prob) > num_pics:
                break

            predictions = torch.zeros(iterations, dtype=torch.int32)
            for _ in range(iterations):
                original_weights = save_model_parameters(model)
                add_noise(model, sigma)

                input = model(x.view(-1, 28*28))
                pred = input.argmax(dim=1, keepdim=True)
                predictions[_] = pred.item()

                restore_model_parameters(model, original_weights)

            entropy = calculate_entropy(predictions)
            probability = torch.bincount(predictions, minlength=10)[
                y.item()]/len(predictions)
            ent_prob.append((entropy, probability))
    return ent_prob


def plot_entropy_prob(ent_prob):
    entropy, probability = zip(*ent_prob)

    plt.figure(figsize=(10, 6))
    plt.scatter(probability, entropy, alpha=0.5)
    plt.xlabel('Probability')
    plt.ylabel('Entropy')
    plt.title('Entropy to Probability')
    plt.grid()
    plt.ylim(min(entropy)-0.1, max(entropy)+0.1)
    plt.xlim(-0.5, 1.5)
    plt.show()


def main():
    n_inputs = 28 * 28
    n_outputs = 10
    train, test = load_data()
    model = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    train_model(model, train, test, optimizer, epochs=1)
    plot_entropy_prob(test_model_noise(
        model, test, 0.1, iterations=10, num_pics=np.inf))


if __name__ == '__main__':
    main()
