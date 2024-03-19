import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from data.loaders import load_SODA, load_MNIST
from models.bayeslens_base import BayesLens
from models.logistic import LogisticRegression
from models.bayeslens_cnn import BayesLensCNN
from models.bayeslens_vit import BayesLens_ViT
from models.vit_b_16 import Pretrained_ViT
from utils.training import train
from utils.perturbation import perturbation, evalute_robustness

writer = SummaryWriter('runs/')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on SODA or MNIST dataset.")
    parser.add_argument('-s', '--soda', action='store_true',
                        help='Use the SODA dataset')
    parser.add_argument('-m', '--mnist', action='store_true',
                        help='Use the MNIST dataset')
    parser.add_argument('-bs', '--batch_size', default=32,
                        type=int, help='Batch size for training and evaluation')
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        help='Number of epochs to train the model')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.soda:
        dataset_path = '../extra/datasets/SODA'
        num_classes = 6
        num_inputs = 3*256*256
        num_channels = 3
    elif args.mnist:
        dataset_path = '../extra/datasets/MNIST'
        num_classes = 10
        num_inputs = 28*28
        num_channels = 1
    else:
        raise ValueError("Either --soda or --mnist must be specified")

    models = [
        # Pretrained_ViT(num_classes=num_classes),
        # BayesLens_ViT(num_classes=num_classes),
        # BayesLensCNN(num_channels=num_channels,
        #              img_size=256 if args.soda else 28, num_classes=num_classes),
        # BayesLens(num_inputs=num_inputs, num_classes=num_classes),
        LogisticRegression(n_inputs=num_inputs, n_outputs=num_classes)
    ]

    for model in models:
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        if args.soda:
            train_loader, val_loader, test_loader = load_SODA(dataset_path, batch_size=args.batch_size, ViT=True if isinstance(
                model, Pretrained_ViT) or isinstance(model, BayesLens_ViT) else False)
        else:
            train_loader, test_loader = load_MNIST(dataset_path, batch_size=args.batch_size, ViT=True if isinstance(
                model, Pretrained_ViT) or isinstance(model, BayesLens_ViT) else False)
            val_loader = test_loader

        train(model, train_loader, val_loader, optimizer,
              epochs=args.epochs, lossfn=criterion, writer=writer)

        perturbation(model, test_loader, sigmas=0.1, iters=10)


if __name__ == "__main__":
    main()
