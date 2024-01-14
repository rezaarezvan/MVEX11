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


def main():
    # Data augmentation and normalization for training
    transform = transforms.Compose([
        # Resize images to fit the input size of ViT
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

# Number of batches to use for training and testing
    num_batches_train = 100
    num_batches_test = 20

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)
    trainloader = itertools.islice(trainloader, num_batches_train)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)
    testloader = itertools.islice(testloader, num_batches_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

# Load a pre-trained Vision Transformer model
    weights = ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)

# Modify the classifier head for CIFAR-10 (10 classes)
    in_features = model.heads[0].in_features
    model.heads[0] = nn.Linear(in_features, 10)

# Move the model to CPU (assuming no GPU available)
    device = torch.device("cpu")
    model.to(device)

# Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training function

    def train_model(model, criterion, optimizer, num_epochs=2):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 50 == 49:  # Print every 50 mini-batches
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0
        print('Finished Training')

# Test function

    def test_model(model, testloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            f'Accuracy of the network on the test images: {100 * correct / total}%')


# Train and test the model
    train_model(model, criterion, optimizer, num_epochs=2)
    test_model(model, testloader)

# Function to show images

    def imshow(img):
        img = img / 2 + 0.5     # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


# In the main function, after training and testing the model

    try:
        # Get some random testing images
        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # Show images
        imshow(torchvision.utils.make_grid(images))
        # Print labels
        print('GroundTruth: ', ' '.join(
            f'{classes[labels[j]]}' for j in range(4)))

        # Predictions
        with torch.no_grad():
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(
            f'{classes[predicted[j]]}' for j in range(4)))

    except StopIteration:
        print("No more data available in the testloader.")


if __name__ == '__main__':
    main()
