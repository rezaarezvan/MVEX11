import sys
import torch
import torch.nn as nn

from data.loaders import load_SODA, load_MNIST
from models.bayeslens_base import BayesLens
from models.logistic import LogisticRegression
from models.bayeslens_cnn import BayesLensCNN
from models.bayeslens_vit import BayesLens_ViT
from models.vit_b_16 import Pretrained_ViT

from utils.helpers import train as train_model
from utils.perturbation import perturbation

SODA = True if '-s' in sys.argv or '--soda' in sys.argv else False
MNIST = True if '-m' in sys.argv or '--mnist' in sys.argv else False
SODA_PATH = '../extra/datasets/SODA'
MNIST_PATH = '../extra/datasets/MNIST'

NUM_INPUTS_VIT = 224*224*3
NUM_INPUTS = 28*28 if MNIST else 3*256*256
NUM_CLASSES = 10 if MNIST else 6
NUM_CHANNELS = 1 if MNIST else 3


def main():
    pretrained_vit = Pretrained_ViT(num_classes=NUM_CLASSES)
    bayeslens_vit = BayesLens_ViT(num_classes=NUM_CLASSES)
    bayeslens_cnn = BayesLensCNN(
        num_channels=NUM_CHANNELS, img_size=256 if SODA else 28, num_classes=NUM_CLASSES)
    bayeslens_base = BayesLens(num_inputs=NUM_INPUTS, num_classes=NUM_CLASSES)
    logistic_regression = LogisticRegression(
        n_inputs=NUM_INPUTS, n_outputs=NUM_CLASSES)

    models = [bayeslens_base]

    for model in models:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        if SODA:
            train, val, test = load_SODA(SODA_PATH, ViT=True if isinstance(
                model, Pretrained_ViT) or isinstance(model, BayesLens_ViT) else False)
        else:
            train, test = load_MNIST(MNIST_PATH, ViT=True if isinstance(
                model, Pretrained_ViT) or isinstance(model, BayesLens_ViT) else False)
            val = test

        train_model(model, train, val, optimizer, epochs=1, lossfn=criterion)

        iterations = 10
        entrop_window_size = 0.1
        sigmas = [0, 0.25, 0.5]
        lambdas = [0.1, 0.5, 1]

        perturbation(model, test, iterations, sigmas,
                     lambdas, entrop_window_size)


if __name__ == "__main__":
    main()
