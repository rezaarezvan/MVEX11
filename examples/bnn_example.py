from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(0, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))

        self.weight_logsigma = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(0, 0.1))
        self.bias_logsigma = nn.Parameter(
            torch.Tensor(out_features).normal_(0, 0.1))

    def forward(self, input):
        weight_dist = dist.Normal(
            self.weight_mu, torch.exp(self.weight_logsigma))
        bias_dist = dist.Normal(self.bias_mu, torch.exp(self.bias_logsigma))
        weight = weight_dist.rsample()
        bias = bias_dist.rsample()
        return F.linear(input, weight, bias)


class BNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BNN, self).__init__()
        self.fc1 = BayesianLinear(input_size, hidden_size)
        self.fc2 = BayesianLinear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('./data', download=True,
                          train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('./data', download=True,
                         train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


def train_bnn(model, trainloader, epochs=50, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(trainloader)}')


bnn_model = BNN(28*28, 256, 10)
train_bnn(bnn_model, trainloader)


def evaluate_bnn(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')


evaluate_bnn(bnn_model, testloader)
