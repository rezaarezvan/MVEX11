import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import itertools
import matplotlib.pyplot as plt
import numpy as np


class ViT:
    def __init__(self, num_classes=10, weights=ViT_B_16_Weights.DEFAULT, device=None):
        self.device = device if device else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.vit_b_16(weights=weights)
        self.model.heads[0] = nn.Linear(
            self.model.heads[0].in_features, num_classes)
        self.model.to(self.device)

    def train(self, trainloader, criterion, optimizer, num_epochs=100):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(
                    self.device), data[1].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 50 == 49:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0
        print('Finished Training')

    def test(self, testloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(
                    self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy}%')
        return accuracy

    def visualize(self, testloader, classes):
        try:
            dataiter = iter(testloader)
            images, labels = next(dataiter)
            imshow(torchvision.utils.make_grid(images))
            print('GroundTruth: ', ' '.join(
                f'{classes[labels[j]]}' for j in range(4)))
            with torch.no_grad():
                outputs = self.model(images.to(self.device))
                _, predicted = torch.max(outputs, 1)
            print('Predicted: ', ' '.join(
                f'{classes[predicted[j]]}' for j in range(4)))
        except StopIteration:
            print("No more data available in the testloader.")


def load_cifar10_dataset(transform, num_batches_train=100, num_batches_test=20):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2)
    trainloader = itertools.islice(trainloader, num_batches_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2)
    testloader = itertools.islice(testloader, num_batches_test)

    return trainloader, testloader


def create_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    transform = create_transform()
    trainloader, testloader = load_cifar10_dataset(transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    vit_model = ViT(num_classes=10, device=None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vit_model.model.parameters(), lr=0.001, momentum=0.9)

    vit_model.train(trainloader, criterion, optimizer, num_epochs=2)
    vit_model.test(testloader)
    vit_model.visualize(testloader, classes)


if __name__ == '__main__':
    main()
