import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from data import load_data
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class BayesLensModelV2(nn.Module):
    def __init__(self, num_classes=6):
        super(BayesLensModelV2, self).__init__()
        self.classifier = BayesianLinear(768, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = self.classifier

    def forward(self, x):
        x = self.vit(x)
        x = self.softmax(x)
        return x


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
    dataset_path = '../extra/datasets/SODA'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train, val, test = load_data(
        dataset_path, batch_size=16, transform=transform)

    model = BayesLensModelV2(num_classes=6)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train, val, optimizer, criterion)


if __name__ == "__main__":
    main()
