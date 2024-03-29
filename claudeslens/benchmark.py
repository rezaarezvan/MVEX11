import os
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from claudeslens.data.loaders import load_SODA, load_MNIST
from claudeslens.models.pretrained_vit import Pretrained_ViT_B_16
from claudeslens.models.claudeslens_vit import ClaudesLens_ViT
from claudeslens.models.claudeslens_convnext import ClaudesLens_ConvNext
from claudeslens.models.pretrained_convnext import Pretrained_ConvNext
from claudeslens.models.claudeslens_logistic import ClaudesLens_Logistic
from claudeslens.utils.training import train, save_model, load_model, add_noise, eval_attention
from claudeslens.utils.perturbation import evalute_weight_perturbation, perturbation, evaluate_robustness

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
                        help='Number of epochs to train the model')
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
                        help='Option to train or just evaluate')
    parser.add_argument('-sp', '--save_plots', action='store_true',
                        help='Save the plots')

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
        pth = './weights/'
        print(f"""Running Model: {model_name}""")
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        if args.soda:
            train_loader, val_loader, test_loader = load_SODA(dataset_path, batch_size=args.batch_size, ViT=True if isinstance(
                model, Pretrained_ViT_B_16) or isinstance(model, ClaudesLens_ViT) else False)
            pth += 'SODA/'
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
            model.eval()
        if args.save_weights:
            save_model(model, pth)
        perturbation(model, test_loader, SAVE_PLOT=args.save_plots)

        # eval_attention(model, test_loader)


if __name__ == "__main__":
    main()
