import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    trainset = torchvision.datasets.MNIST(
        root='../extra/datasets', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(
        root='../extra/datasets', train=False, download=True, transform=transforms.ToTensor())

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader
torch.manual_seed(0)
np.random.seed(0)

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.softmax(x, dim=1)
        return x


def train_model(model, train, test, optimizer, epochs=3):
    loss = nn.CrossEntropyLoss()
    acc = 0

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train):
            optimizer.zero_grad()
            input = model(data.view(-1, 28 * 28))
            output = loss(input, target)
            output.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {i}, Loss: {output.item()}')

        print(f'Epoch {epoch + 1} completed')
        print('Testing model...')
        acc = test_model(model, test)
        print('')

    return acc


def test_model(model, test):
    loss = nn.CrossEntropyLoss()

    model.eval()

    test_loss = 0
    correct = 0

    certainty = [0] * 10

    with torch.no_grad():
        for data, target in test:
            input = model(data.view(-1, 28 * 28))

            for i, true_label in enumerate(target):
                p = input[i, true_label].item()
                idx = round(p * 10) - 1
                if idx < 0:
                    idx = 0
                certainty[idx] += 1

            test_loss += loss(input, target).item()
            pred = input.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test.dataset)
    acc = 100. * correct / len(test.dataset)
    print(
        f'Average loss: {test_loss: .4f}, Accuracy: {correct}/{len(test.dataset)} ({acc: .2f} %)')
    print(f'Certainty: {certainty}')

    return acc


def save_model_parameters(model):
    original_params = [param.clone() for param in model.parameters()]
    return original_params


def restore_model_parameters(model, original_params):
    with torch.no_grad():
        for param, original in zip(model.parameters(), original_params):
            param.copy_(original)


def add_noise(model, sigma):
    with torch.no_grad():
        for param in model.parameters():
            param += torch.randn_like(param) * sigma
    return model


def calculate_entropy(predictions: []):
    value_counts = np.bincount(predictions) / len(predictions)
    entropy = -np.sum([p * np.log(p) for p in value_counts if p > 0])
    return entropy


def evaluate_model_with_noise(model, image, label, sigma, iterations):
    predictions = []
    original_params = save_model_parameters(model)  # Save original model parameters
    for _ in range(iterations):
        add_noise(model, sigma)  # Add noise to model
        output = model(image.view(-1, 28 * 28))  # Use the modified model directly
        pred = output.argmax(dim=1, keepdim=True).item()
        predictions.append(pred)
        restore_model_parameters(model, original_params)  # Restore original parameters after each iteration
    entropy = calculate_entropy(predictions)
    correct_predictions = predictions.count(label)
    probability = correct_predictions / iterations
    return entropy, probability


def noise_acc_plot(model, test_loader, sigma, iterations):
    model.eval()
    results = []
    with torch.no_grad():
        for data, target in test_loader:
            for i in range(len(data)):
                image, label = data[i].unsqueeze(0), int(target[i])
                entropy, probability = evaluate_model_with_noise(model, image, label, sigma, iterations)
                results.append((entropy, probability))
    print(results)
    plot_results(results, sigma)


def main():
    n_inputs = 28 * 28
    n_outputs = 10
    train, test = load_data()
    model = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    no_noise_acc = train_model(model, train, test, optimizer, epochs=0)

    noise_acc_plot(model, test, sigma=0, iterations=10)


#  print(f'Non-noise accuracy: {no_noise_acc}')
#  print(f'Noise accuracy: {noise_acc}')
#  print(f'Accuracy difference: {no_noise_acc - noise_acc}')

def plot_results(results, sigma):
    entropies, probabilities = zip(*results)
    plt.scatter(entropies, probabilities, color='blue')
    plt.title(f'Entropy vs. Probability with sigma = {sigma}')
    plt.xlabel('Entropy')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
