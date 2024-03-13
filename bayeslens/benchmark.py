import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data import load_SODA, load_MNIST
from models.bayeslens_base import BayesLens
from models.bayeslens_cnn import BayesLensCNN
from models.bayeslens_vit import BayesLens_ViT
from models.vit_b_16 import Pretrained_ViT
from utils.model import train_model


def main():
    dataset_path_SODA = '../extra/datasets/SODA'
    dataset_path_MNIST = '../extra/datasets/MNIST'

    transform_SODA = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # train, val, test = load_SODA(
    #     dataset_path_SODA, batch_size=16, transform=transform_SODA)

    train, test = load_MNIST(dataset_path_MNIST)
    val = test

    pretrained_vit = Pretrained_ViT()
    bayeslens_vit = BayesLens_ViT()
    bayeslens_cnn = BayesLensCNN()
    bayeslens_base = BayesLens(28*28, num_classes=10)

    models = [pretrained_vit, bayeslens_vit, bayeslens_cnn, bayeslens_base]

    optimizers = [torch.optim.Adam(
        model.parameters(), lr=0.1, weight_decay=1e-4) for model in models]
    criterion = nn.CrossEntropyLoss()

    train_model(models[3], train, val, test,
                optimizers[3], criterion, num_epochs=1)

    # iterations = 2
    # entropies = []
    # weighted_average = []
    #
    # sigmas = np.arange(0.1, 1, 0.1)
    # sigmas = np.append(sigmas, [1, 10, 100, 500])
    #
    # for sigma in sigmas:
    #     print(f"Sigma: {sigma}")
    #     entropy = test_model_noise(model, test, sigma=sigma, iters=iterations)
    #     weighted_average.append((calculate_weighted_averages(entropy), sigma))
    #     entropies.append(entropy)
    #     for lam in [0.1, 0.5, 1]:
    #         print(f"Lambda: {lam}, K: {compute_k(entropy, _lambda=lam)}")
    #     print('-----------------------------------\n')
    #
    # plot_weighted_averages(weighted_average)
    # for entropy, sigma in zip(entropies, sigmas):
    #     plot_entropy_prob(entropy, sigma, 10,
    #                       iterations)


if __name__ == "__main__":
    main()
