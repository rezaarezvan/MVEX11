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
from utils.bench import test_model_noise
from utils.entrop import get_weight_avg, get_psi, get_best_sigma
from utils.plot import plot_entropy_prob, plot_weight_avg

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
    bayeslens_base = BayesLens(256*256, num_classes=6)

    models = [bayeslens_cnn]

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

        iterations = 10
        entrop_window_size = 0.1

        entropies = []
        weighted_average = []

        psi_list = []
        sigmas = [0, 0.25, 0.5]
        for sigma in sigmas:
            print(f"Sigma: {sigma}")
            entropy = test_model_noise(
                model, test, sigma=sigma, iters=iterations)
            weighted_average.append(
                (get_weight_avg(entropy, window_size=entrop_window_size), sigma))
            entropies.append(entropy)
            for lam in [0.1, 0.5, 1]:
                print(f"Lambda: {lam}, K: {get_psi(entropy, _lambda=lam)}")
            psi_list.append(get_psi(entropy))
            print('-----------------------------------\n')

        print(get_best_sigma(psi_list, sigmas))
        plot_weight_avg(weighted_average, )
        for entropy, sigma in zip(entropies, sigmas):
            plot_entropy_prob(entropy, sigma, 0, iterations)


if __name__ == "__main__":
    main()
