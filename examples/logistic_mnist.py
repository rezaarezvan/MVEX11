import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


def load_data():
    train = torchvision.datasets.MNIST(root='../extra/datasets', train=True, download=True, transform=transforms.ToTensor())
    test  = torchvision.datasets.MNIST(root='../extra/datasets', train=False, download=True, transform=transforms.ToTensor())

    train = DataLoader(train, batch_size=32, shuffle=True)
    test  = DataLoader(test, batch_size=32, shuffle=False)

    return train, test


class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


def train_model(model, train, test, optimizer, loss, epochs=3):

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train):

            # Forward pass
            optimizer.zero_grad()
            input = model(data.view(-1, 28*28))
            output = loss(input, target)

            # Backward pass
            output.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {output.item()}')

        print(f'Epoch {epoch+1} completed')
        print('Testing model...')
        test_model(model, test, loss)
        print('')


def test_model(model, test, loss):
    model.eval()

    test_loss = 0
    correct   = 0

    with torch.no_grad():
        for data, target in test:
            input = model(data.view(-1, 28*28))
            pred  = input.argmax(dim=1, keepdim=True)

            test_loss += loss(input, target).item()
            correct   += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test.dataset)
    acc        = 100. * correct / len(test.dataset)

    print(f'Average loss: {test_loss: .4f}, Accuracy: {correct}/{len(test.dataset)} ({acc: .2f} %)')


def main():
    n_inputs  = 28 * 28
    n_outputs = 10

    train, test = load_data()

    model     = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss      = nn.CrossEntropyLoss()

    train_model(model, train, test, optimizer, loss, epochs=10)


if __name__ == '__main__':
    main()
