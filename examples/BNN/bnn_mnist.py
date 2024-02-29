import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='../extra/datasets', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.MNIST(root='../extra/datasets', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader  = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu       = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.01))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-5))

        self.bias_mu       = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.01))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features).fill_(-5))

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma   = torch.exp(self.bias_logsigma)

        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias   = self.bias_mu   + bias_sigma   * torch.randn_like(self.bias_mu)

        return F.linear(x, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.fc1 = BayesianLinear(28*28, 400)
        self.fc2 = BayesianLinear(400, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def train_model(model, train, test, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train):
            optimizer.zero_grad()
            output = model(data.view(-1, 28*28))
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}')

        print(f'Epoch {epoch+1} completed')
        print('Testing model...')
        test_model(model, test)
        print('')


def test_model(model, test):
    model.eval()

    test_loss = 0
    correct   = 0

    with torch.no_grad():
        for data, target in test:
            output     = model(data.view(-1, 28*28))
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred       = output.argmax(dim=1, keepdim=True)
            correct   += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test.dataset)
    acc = 100. * correct / len(test.dataset)
    print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test.dataset)} ({acc:.2f}%)')


def main():
    train, test = load_data()
    model       = BayesianNetwork()
    optimizer   = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train, test, optimizer, epochs=10)


if __name__ == '__main__':
    main()
