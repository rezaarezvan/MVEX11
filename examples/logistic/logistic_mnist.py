import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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


def main():
    n_inputs = 28 * 28
    n_outputs = 10

    train, test = load_data()

    model = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss = nn.CrossEntropyLoss()

    train_model(model, train, test, optimizer, loss, epochs=10)


if __name__ == '__main__':
    main()
