import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='../extra/datasets', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='../extra/datasets', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader


'''
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        # Assuming input is (batch_size, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)  # Output: (batch_size, 10, 28, 28)
        self.conv2 = nn.Conv2d(10, 1, kernel_size=3, padding=1)  # Output: (batch_size, 1, 28, 28)

    def forward(self, x):
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))  # Get attention weights in [0,1]
        return x * attention  # Apply attention by element-wise multiplication
'''


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear spaces
        self.to_query = nn.Linear(input_dim, hidden_dim)
        self.to_key = nn.Linear(input_dim, hidden_dim)
        self.to_value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        Q = self.to_query(x)
        K = self.to_key(x)
        V = self.to_value(x)
        # Attention formula straight off
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, V)

        return out


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.001))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-5))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.001))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features).fill_(-5))

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)

        return F.linear(x, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()
        hidden_dim = 28 * 3
        input_size = 28 ** 2
        features = 400
        labels = 10
        self.attention = AttentionLayer(input_size, hidden_dim)
        self.projection = nn.Linear(hidden_dim, input_size)
        self.fc1 = BayesianLinear(input_size, features)
        self.fc2 = BayesianLinear(features, labels)

    def forward(self, x):
        x_flattened = x.view(x.size(0), -1)
        attention_out = self.attention(x_flattened)

        x_projected = self.projection(attention_out)

        x = F.relu(self.fc1(x_projected))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


def test_model(model, test):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test:
            output = model(data.view(-1, 28 * 28))
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test.dataset)
    acc = 100. * correct / len(test.dataset)
    print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test.dataset)} ({acc:.2f}%)')


def train_model(model, train, test, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train):
            optimizer.zero_grad()
            output = model(data.view(-1, 28 * 28))
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}')

        print(f'Epoch {epoch + 1} completed')
        print('Testing model...')
        test_model(model, test)
        print('')


'''

def test_model(model, test):
    model.eval()
    test_loss = 0
    correct = 0
    # Select a small number of test images to visualize
    num_images_to_visualize = 5

    with torch.no_grad():
        vis = True
        for data, target in test:
            output, attention_map = model(data.view(-1, 28 * 28), return_attention=True)  # Request attention maps
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Visualization
            if vis:
                for i in range(num_images_to_visualize):
                    plt.subplot(2, num_images_to_visualize, i + 1)
                    plt.imshow(data[i].view(28, 28).cpu().numpy(), cmap='gray')
                    plt.title("Original")
                    plt.axis('off')

                    plt.subplot(2, num_images_to_visualize, num_images_to_visualize + i + 1)
                    attention_img = attention_map[i].squeeze().cpu().numpy()  # Squeeze to remove channel dim
                    plt.imshow(attention_img, cmap='spring', interpolation='nearest')
                    plt.title("Attention")
                    plt.axis('off')

                plt.show()
                vis = False
                # break  # Only visualize for the first batch

    test_loss /= len(test.dataset)
    acc = 100. * correct / len(test.dataset)
    print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test.dataset)} ({acc:.2f}%)')
'''


def main():
    train, test = load_data()
    model = BayesianNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train, test, optimizer, epochs=10)


if __name__ == '__main__':
    main()
