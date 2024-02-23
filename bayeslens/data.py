import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_gtsrb(batch_size=32, download=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.GTSRB(
        root='../extra/datasets', split='train', download=download, transform=transform)

    test_dataset = datasets.GTSRB(
        root='../extra/datasets', split='test', download=download, transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_gtsrb()
    print("Train and test loaders have been successfully created.")
