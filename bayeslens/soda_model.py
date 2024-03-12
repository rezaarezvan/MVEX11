import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
from soda import load_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
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
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(0, 0.1))
        self.weight_sigma = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(0, 0.1))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_sigma = nn.Parameter(
            torch.Tensor(out_features).normal_(0, 0.1))

    def forward(self, x):
        weight = Normal(self.weight_mu, F.softplus(
            self.weight_sigma)).rsample()
        bias = Normal(self.bias_mu, F.softplus(self.bias_sigma)).rsample()
        return F.linear(x, weight, bias)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-2)
        self.last_attention = None

    def forward(self, x):
        query = self.query_conv(x).view(
            x.shape[0], -1, x.shape[2]*x.shape[3]).permute(0, 2, 1)
        key = self.key_conv(x).view(x.shape[0], -1, x.shape[2]*x.shape[3])
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
    def __init__(self, num_classes, dropout_rate=0.5):
        super(BayesLensModel, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            SelfAttention(32),
        )
        self.flatten = nn.Flatten()
        self.classifier = BayesianLinear(32*64*64, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.dropout(x)
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


def train_model(model, train, test, optimizer, criterion, num_epochs=40):
    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train)}], Loss: {loss.item():.4f}')

        evaluate_model(model, test)
        print()


def evaluate_model(model, test, mode='validation'):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'{mode.capitalize()} accuracy of the model: {accuracy} %')


def main():
    dataset_path = '../extra/datasets/labeled_trainval'
    train, val = load_data(dataset_path, batch_size=16)

    model = BayesLensModel(num_classes=6)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train, val, optimizer, criterion)


if __name__ == "__main__":
    main()