from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_gtsrb(batch_size=32, download=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ])

    train_dataset = datasets.GTSRB(
        root='../extra/datasets', split='train', download=download, transform=transform)

    val_dataset = datasets.GTSRB(
        root='../extra/datasets', split='train', download=download, transform=transform)

    test_dataset = datasets.GTSRB(
        root='../extra/datasets', split='test', download=download, transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_gtsrb()
    print(len(train_loader), len(val_loader), len(test_loader))
