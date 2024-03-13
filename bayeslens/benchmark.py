import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data import load_SODA, load_MNIST
from models.bayeslens_base import BayesLens
from models.bayeslens_cnn import BayesLensCNN
from models.bayeslens_vit import BayesLens_ViT
from models.vit_b_16 import Pretrained_ViT
from utils.model import train_model

SODA = True if '-s' in sys.argv or '--soda' in sys.argv else False
MNIST = True if '-m' in sys.argv or '--mnist' in sys.argv else False
SODA_PATH = '../extra/datasets/SODA'
MNIST_PATH = '../extra/datasets/MNIST'
TRANSFORM_VIT = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def main():
    pretrained_vit = Pretrained_ViT()
    bayeslens_vit = BayesLens_ViT()
    bayeslens_cnn = BayesLensCNN()
    bayeslens_base = BayesLens(28*28, num_classes=10)

    models = [pretrained_vit, bayeslens_vit, bayeslens_cnn, bayeslens_base]

    optimizers = [torch.optim.Adam(
        model.parameters(), lr=0.1, weight_decay=1e-4) for model in models]
    criterion = nn.CrossEntropyLoss()

    for model in models:
        transform = TRANSFORM_VIT if isinstance(
            model, Pretrained_ViT) or isinstance(model, BayesLens_ViT) else None

        if SODA:
            train, val, test = load_SODA(SODA_PATH, transform=transform) if transform else load_SODA(
                SODA_PATH)
        else:
            train, test = load_MNIST(MNIST_PATH, transform=transform) if transform else load_MNIST(
                MNIST_PATH)
            val = None

        if isinstance(model, BayesLens):
            train_model(model, train, val, test, optimizers[models.index(
                model)], criterion, flatten=True, num_epochs=1)
        else:
            train_model(model, train, val, test, optimizers[models.index(
                model)], criterion, num_epochs=1)


if __name__ == "__main__":
    main()
