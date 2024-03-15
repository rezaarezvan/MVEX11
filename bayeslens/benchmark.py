import sys
import torch
import torch.nn as nn

from data import load_SODA, load_MNIST
from models.bayeslens_base import BayesLens
from models.bayeslens_cnn import BayesLensCNN
from models.bayeslens_vit import BayesLens_ViT
from models.vit_b_16 import Pretrained_ViT

from utils.helpers import train_model
from utils.perturbation import test_model_noise
from utils.metrics import weight_avg, psi, best_sigma
from utils.plot import plot_entropy_prob, plot_weight_avg

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
        num_inputs=NUM_CHANNELS, img_size=256 if SODA else 28, num_classes=NUM_CLASSES)
    bayeslens_base = BayesLens(num_inputs=NUM_INPUTS, num_classes=NUM_CLASSES)

    models = [pretrained_vit, bayeslens_vit, bayeslens_cnn, bayeslens_base]

    optimizers = [torch.optim.Adam(
        model.parameters(), lr=0.1, weight_decay=1e-4) for model in models]
    criterion = nn.CrossEntropyLoss()

    for model in models:
        if SODA:
            train, val, test = load_SODA(SODA_PATH, ViT=True) if isinstance(
                model, Pretrained_ViT) or isinstance(model, BayesLens_ViT) else load_SODA(SODA_PATH)
        else:
            train, test = load_MNIST(MNIST_PATH, ViT=True) if isinstance(
                model, Pretrained_ViT) or isinstance(model, BayesLens_ViT) else load_MNIST(MNIST_PATH)
            val = test

        if isinstance(model, BayesLens):
            train_model(model, train, val, test, optimizers[models.index(
                model)], criterion, flatten=True, num_epochs=1)
        else:
            train_model(model, train, val, test, optimizers[models.index(
                model)], criterion, num_epochs=1)

        iterations = 10
        entrop_window_size = 0.1

        entropies = []
        weighted_average = []

        psi_list = []
        sigmas = [0, 0.25, 0.5]
        for sigma in sigmas:
            print(f"σ: {sigma}")
            entropy = test_model_noise(
                model, test, sigma=sigma, iters=iterations, CNN=False)
            weighted_average.append(
                (weight_avg(entropy, window_size=entrop_window_size), sigma))
            entropies.append(entropy)
            for lam in [0.1, 0.5, 1]:
                print(f"λ: {lam}, ψ: {psi(entropy, _lambda=lam)}")
            psi_list.append(psi(entropy))
            print('-----------------------------------\n')

        print(best_sigma(psi_list, sigmas))
        plot_weight_avg(weighted_average, )
        for entropy, sigma in zip(entropies, sigmas):
            plot_entropy_prob(entropy, sigma, 0, iterations)


if __name__ == "__main__":
    main()
