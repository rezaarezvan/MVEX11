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
from utils.training import train, save_model, load_model
from utils.perturbation import perturbation, evalute_robustness

writer = SummaryWriter('runs/')

model_choices = {
        'log': LogisticRegression,
        'pv' : Pretrained_ViT,
        'bv' : BayesLens_ViT,
        'bc' : BayesLensCNN,
        'b'  : BayesLens
        }

def list_of_models(arg):
    print(arg)
    return arg.split(',')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on SODA or MNIST dataset.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--soda', action='store_true',
                        help='Use the SODA dataset')
    parser.add_argument('-m', '--mnist', action='store_true',
                        help='Use the MNIST dataset')
    parser.add_argument('-bs', '--batch_size', default=32,
                        type=int, help='Batch size for training and evaluation')
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        help='Number of epochs to train the model')
    parser.add_argument('-im', '--init_models', type=list_of_models, default=['log'],
                        help="Different Models to test/train\n"
                        "pv  = Pretrained_ViT\n"
                        "bv  = BayesLens_ViT\n"
                        "bc  = BayesLensCNN\n"
                        "b   = BayesLens\n"
                        "log = LogisticRegression")
    parser.add_argument('-lw', '--load_weights', action='store_true',
                        help='Load weights for initalized models')
    parser.add_argument('-sw', '--save_weights', action='store_true',
                        help='Saves the models after training in their respective .pth')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Option to train or just evaluate')

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

    if args.init_models:
        models = []
        for model in args.init_models:
            model = model_choices[model](num_channels=num_channels,
                                         num_inputs=num_inputs, 
                                         num_classes=num_classes)
            models.append(model)


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

        if args.train and not args.load_weights:
            train(model, train_loader, val_loader, optimizer,
                  epochs=args.epochs, lossfn=criterion, writer=writer)
        if args.load_weights:
            load_model(model, f'models/model_dirs/{model.__class__.__name__}.pth')
            model.eval()
        if args.save_weights:
            save_model(model, f'models/model_dirs/{model.__class__.__name__}.pth')
            

if __name__ == "__main__":
    main()
