import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='../extra/datasets', train=True,  transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='../extra/datasets', train=False, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

class BayesianLinear(nn.Module):
    """Bayesian linear layer."""
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters for the mean of weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        # Parameters for the variance of weights (log sigma)
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        
        # Parameters for the mean of biases
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        # Parameters for the variance of biases (log sigma)
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))

    def forward(self, x):
        # Sample weights and biases from their respective distributions
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.hidden = BayesianLinear(28*28, 1200)  # Example size
        self.out = BayesianLinear(1200, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = F.relu(self.hidden(x))
        x = F.log_softmax(self.out(x), dim=1)  # Use log-softmax for the output
        return x

def train(model, trainloader, testloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for data, target in trainloader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testloader.dataset)
        print(f'Epoch: {epoch+1}, Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({100. * correct / len(testloader.dataset):.0f}%)')

model = BayesianNetwork()
train(model, trainloader, testloader, epochs=10)

