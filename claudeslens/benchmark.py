import os
import sys
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from claudeslens.utils.perturbation import perturbation
from claudeslens.data.loaders import load_SODA, load_MNIST
from claudeslens.utils.training import train, save_model, load_model
from claudeslens.models import Pretrained_ViT_B_16, ClaudesLens_ViT, Pretrained_ConvNext, ClaudesLens_ConvNext, ClaudesLens_Logistic

old_stdout = sys.stdout

writer = SummaryWriter('runs/')

model_choices = {
    'pv': Pretrained_ViT_B_16,
    'cv': ClaudesLens_ViT,
    'pc': Pretrained_ConvNext,
    'cc': ClaudesLens_ConvNext,
    'cl': ClaudesLens_Logistic,
}


def list_of_models(arg):
    """
    Converts init_models argument from CLI from format
    "model1,model2,model3" to ["model1", "model2", "model3"]
    """
    return arg.split(',')


def parse_arguments():
    """
    Parse the arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on SODA or MNIST dataset.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--soda', action='store_true',
                        help='Using SODA dataset instead of MNIST')
    parser.add_argument('-bs', '--batch_size', default=32,
                        type=int, help='Batch size for training and evaluation')
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        help='Number of epochs to train the models')
    parser.add_argument('-m', '--models', type=list_of_models, default=['log'],
                        help="Different Models to test/train\n"
                        "pv  = Pretrained_ViT\n"
                        "cv  = ClaudesLens_ViT\n"
                        "pc  = PretrainedConvNext\n"
                        "cc  = ClaudesLens_ConvNext\n"
                        "cl  = ClaudesLens_Logistic\n"
                        "Example: -m pv,cv,pc,cc,cl")
    parser.add_argument('-lw', '--load_weights', action='store_true',
                        help='Load weights for initalized models')
    parser.add_argument('-sw', '--save_weights', action='store_true',
                        help='Saves the models after training in their respective .pth')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Option to train the models')
    parser.add_argument('-sp', '--save_plots', action='store_true',
                        help='Save the plots')
    parser.add_argument('-b', '--benchmark', action='store_true',
                        help='Option to benchmark the models')
    parser.add_argument('-l', '--log', action='store_true',
                        help='Option to log the output to a file')
    parser.add_argument('-ld', '--load_data', action='store_true',
                        help='Load data for models')

    args = parser.parse_args()
    return args


def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(
        current_dir, '..') if 'claudeslens' in current_dir else current_dir
    return os.path.abspath(project_root)


def get_dataset_path(dataset_name):
    return os.path.join(get_project_root(), 'extra', 'datasets', dataset_name)


def main():
    args = parse_arguments()
    dataset_name = 'SODA' if args.soda else 'MNIST'
    if args.soda:
        dataset_path = get_dataset_path('SODA')
        num_classes = 6
        num_inputs = 3*256*256
        num_channels = 3
    else:
        dataset_path = get_dataset_path('MNIST')
        num_classes = 10
        num_inputs = 28*28
        num_channels = 1

    models = []
    for model in args.models:
        if model not in model_choices:
            print(f"Model {model} not found in model_choices")
            continue

        model = model_choices[model](num_channels=num_channels,
                                     num_inputs=num_inputs,
                                     num_classes=num_classes)
        models.append(model)

    for model in models:
        model_name = model.__class__.__name__
        os.makedirs(f'logs/{dataset_name}', exist_ok=True)
        log_file = open(f'logs/{dataset_name}/{model_name}.log', 'a')
        sys.stdout = log_file if args.log else old_stdout
        pth = './weights/'
        print(f"""Running Model: {model_name}""")
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        if args.soda:
            train_loader, val_loader, test_loader = load_SODA(dataset_path, batch_size=args.batch_size, ViT=True if isinstance(
                model, Pretrained_ViT_B_16) or isinstance(model, ClaudesLens_ViT) else False)
            pth += 'SODA'
        else:
            train_loader, test_loader = load_MNIST(dataset_path, batch_size=args.batch_size, ViT=True if isinstance(model, Pretrained_ViT_B_16) or isinstance(
                model, ClaudesLens_ViT) else False, ConvNext=True if isinstance(model, Pretrained_ConvNext) or isinstance(model, ClaudesLens_ConvNext) else False)
            val_loader = test_loader
            pth += 'MNIST'

        pth += f'/{model_name}.pth'

        if args.train and not args.load_weights:
            train(model, train_loader, val_loader, optimizer,
                  epochs=args.epochs, lossfn=criterion, writer=writer)
        if args.load_weights:
            load_model(model, pth)
        if args.save_weights:
            save_model(model, pth)
        if args.benchmark:
            sigmas = [0, 0.1, 0.5, 1, 10]
            lambdas = [0.1, 0.5, 1, 2]
            perturbation(model, test_loader, sigmas=sigmas,
                         lambdas=lambdas, SAVE_PLOT=args.save_plots, LOAD_DATA=args.load_data)

        log_file.close()


if __name__ == "__main__":
    main()
