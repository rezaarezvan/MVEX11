import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train)}], Loss: {loss.item():.4f}')

        print(f'Epoch {epoch+1} completed')
        print('Testing model...')
        test_model(model, test)
        print()


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


@torch.no_grad()
def test_model_noise(model, test, sigma, iterations=10):
    model.eval()
    ent_prob = []

    for batch, (images, labels) in enumerate(test):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        batch_size = images.size(0)

        entropy_cum = torch.zeros(batch_size, device=DEVICE)
        correct_prob_cum = torch.zeros(batch_size, device=DEVICE)

        for i in range(iterations):
            original_weights = save_model_parameters(model)
            add_noise(model, sigma)

            outputs = model(images.flatten(start_dim=1))
            probs = nn.functional.softmax(outputs, dim=1)

            for j in range(batch_size):
                entropy_cum[j] += Categorical(probs=probs[j]).entropy()
                correct_prob_cum[j] += probs[j, labels[j]]

            restore_model_parameters(model, original_weights)

        entropy_cum /= iterations
        correct_prob_cum /= iterations

        for i in range(batch_size):
            ent_prob.append(
                (entropy_cum[i].item(), correct_prob_cum[i].item()))

        if (batch+1) % 50 == 0:
            print(f'Batch [{batch+1}/{len(test)}] completed')

    return ent_prob


def plot_entropy_prob(ent_prob):
    entropy, probability = zip(*ent_prob)
    plt.figure(figsize=(10, 6))
    plt.scatter(probability, entropy, alpha=0.5, s=1)
    plt.xlabel('Correct Probability')
    plt.ylabel('Entropy')
    plt.title('Entropy vs Correct Probability')
    plt.grid()
    plt.ylim(min(entropy)-0.1, max(entropy)+0.1)
    plt.xlim(-0.5, 1.5)
    plt.show()

def prob_ent_window_plot(ent_prob):

    value_list = []
    entropy, probability = zip(*ent_prob)
    lower_off = 0.0
    higher_off = 0.25
    for ent_range in range(8):
        limited_list = list()
        for entropy in ent_prob:
            if higher_off >= entropy[0] >= lower_off:
                limited_list.append(entropy)
        sum = 0
        print(ent_range)
        for prob in limited_list:
            sum += prob[1]
        value = sum / len(limited_list)
        print(value)
        value_list.append(value)
        print("New iteration:")
        lower_off += 0.25
        higher_off += 0.25



    plt.figure(figsize=(10, 6))
    plt.plot(value_list, marker='o')
    plt.xlabel('Correct Probability')
    plt.ylabel('Entropy')
    plt.title('Entropy vs Correct Probability')
    plt.grid()
    plt.show()

def main():
    n_inputs = 28 * 28
    n_outputs = 10
    train, test = load_data()
    model = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train, test, optimizer, criterion, epochs=1)
    plot_entropy_prob(test_model_noise(model, test, sigma=0.1, iterations=10))


if __name__ == '__main__':
    main()
