import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
from data import load_gtsrb


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))

        self.bias_mu      = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_sigma   = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))

    def forward(self, x):
        weight = Normal(self.weight_mu, F.softplus(self.weight_sigma)).rsample()
        bias   = Normal(self.bias_mu, F.softplus(self.bias_sigma)).rsample()
        return F.linear(x, weight, bias)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.softmax        = nn.Softmax(dim=-2)
        self.last_attention = None

    def forward(self, x):
        query = self.query_conv(x).view(x.shape[0], -1, x.shape[2]*x.shape[3]).permute(0, 2, 1)
        key   = self.key_conv(x).view(x.shape[0], -1, x.shape[2]*x.shape[3])
        value = self.value_conv(x).view(x.shape[0], -1, x.shape[2]*x.shape[3])

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(x.shape)
        self.last_attention = attention
        return out + x

    def get_last_attention(self):
        return self.last_attention


class BayesLensModel(nn.Module):
    def __init__(self):
        super(BayesLensModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            SelfAttention(32),
        )
        self.flatten    = nn.Flatten()
        self.classifier = BayesianLinear(32 * 8 * 8, 43)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def forward_attention(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        attention_scores = self.feature_extractor[-1].get_last_attention()
        return x, attention_scores


def visualize_attention(original_image, attention_scores):
    image = original_image.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    attention = attention_scores.sum(dim=1)[0]
    attention = attention.cpu().detach().numpy()
    attention_resized = np.resize(attention, (image.shape[0], image.shape[1]))

    attention_resized = (attention_resized - np.percentile(attention_resized, 10)) / \
        (np.percentile(attention_resized, 90) -
         np.percentile(attention_resized, 10))
    attention_resized = np.clip(attention_resized, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    plt.imshow(attention_resized, cmap='jet', alpha=0.4)
    plt.title("Attention Overlay")
    plt.axis('off')
    plt.show()


def train(model, train_loader, optimizer, criterion, epochs=10, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')


def test(model, test_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    correct    = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output      = model(data)
            total_loss += criterion(output, target).item()
            pred        = output.argmax(dim=1, keepdim=True)
            correct    += pred.eq(target.view_as(pred)).sum().item()
    print(f'Test Loss: {total_loss/len(test_loader)}, Accuracy: {100. * correct / len(test_loader.dataset)}%')


def main():
    train_loader, test_loader = load_gtsrb(batch_size=32)
    model = BayesLensModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, criterion, epochs=10)
    test(model, test_loader, criterion)

    # Visualize attention
    model.eval()
    data, _ = next(iter(test_loader))
    _, attention = model.forward_attention(data)
    for i in range(5):
        visualize_attention(data[i], attention)


if __name__ == "__main__":
    main()
