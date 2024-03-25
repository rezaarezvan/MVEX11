import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .metrics import entropy, weight_avg, psi, best_sigma
from .training import add_noise, restore_parameters, save_parameters, evaluate
from .plot import plot_entropy_acc_cert, plot_weight_avg, plot_most_often_similar

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

"""
Everything calculated in this file is done batch-wise.
"""
@torch.no_grad()
def evaluate_robustness(model, test_loader, sigmas=[0.1, 0.25, 0.5], iters=10):
    """
    Evaluate the robustness of the model by adding noise to the model parameters
    and evaluating the models accuracy on the entire test set
    """
    model.eval()
    model.to(DEVICE)
    result = []
    loop = tqdm(sigmas, leave=False, disable=False)
    for sigma in loop:
        tmp = []
        for _ in range(iters):
            original_params = save_parameters(model)
            add_noise(model, sigma)

            acc = evaluate(model, test_loader)
            tmp.append(acc)

            restore_parameters(model, original_params)

        acc_mean = sum(tmp) / len(tmp)
        result.append((sigma, acc_mean))
    return result


@torch.no_grad()
def evaluateMixUp(model, test_loader, sigma=0.3, iterations=20):
    """
    Computes to what degree the model mixes up the label (Calculates the entropy
    without the given correct label)
    Args:
        sigma has default parameter of 0.01
        iterations has default parameter of 10
    Returns:
        A collection of all the predictions with the correct label
        [((length of predictions), correct label), ...]
        [([1,2,0,1,1,2,1,1,1], 1), ...]
    """

    model.eval()
    model.to(DEVICE)
    loop = tqdm(test_loader, leave=False)
    for images, labels in loop:
        predictions = []
        probs = []
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        for _ in range(iterations):

            original_params = save_parameters(model)
            add_noise(model, sigma)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            prob = nn.functional.softmax(outputs, dim=1)
            correct_class_probs = prob.gather(1, labels.unsqueeze(1)).squeeze()
            predictions.append(preds)
            probs.append(correct_class_probs)

            restore_parameters(model, original_params)

        all_predictions = torch.stack(predictions, dim=1)

    # List comprehension adding together each element into their specific row
    # This will sort the list as following:
    # [((length of predictions), correct label), ...]
    #
    matrix_with_correct_label = [(all_predictions[i], labels[i].item()) for i in range(len(labels))]



    plot_most_often_similar(matrix_with_correct_label, 0, 2)



@torch.no_grad()
def evalute_perturbation(model, test_loader, sigma=0, iters=10):
    """
    Evaluate the perturbation of the model by adding noise to the model parameters
    and calculating the entropy, accuracy and certainty of the model for each batch
    """
    model.eval()
    model.to(DEVICE)
    result = []
    total_pred = []
    all_labels = []

    loop = tqdm(test_loader, leave=False)
    for images, labels in loop:
        predictions = []
        probs = []
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        for _ in range(iters):
            original_params = save_parameters(model)
            add_noise(model, sigma)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            prob = nn.functional.softmax(outputs, dim=1)
            correct_class_probs = prob.gather(1, labels.unsqueeze(1)).squeeze()

            predictions.append(preds)
            probs.append(correct_class_probs)

            restore_parameters(model, original_params)

        pred_matrix = torch.stack(predictions, dim=1)
        prob = torch.stack(probs).mean(dim=0)
        ent = entropy(pred_matrix)
        accuracy = (pred_matrix == labels.view(-1, 1)).float().mean(dim=1)

        for e, a, c in zip(ent, accuracy, prob):
            result.append((e.item(), a.item(), c.item()))

    return result


def perturbation(model, test_loader, iters=20, sigmas=[0.01], lambdas=[0.1, 0.5, 1], entropy_window_size=0.1, SAVE_PLOT=True):
    """
    Main evaluation loop for the perturbation tests for the model
    """
    entropies = []
    weighted_average = []
    psi_list = []

    for sigma in sigmas:
        print(f"σ: {sigma}")
        entropy = evalute_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        weighted_average.append(
            (weight_avg(entropy, window_size=entropy_window_size), sigma))
        entropies.append(entropy)
        for _lambda in lambdas:
            print(f"λ: {_lambda}, ψ: {psi(entropy, _lambda=_lambda)}")
        psi_list.append(psi(entropy))
        print('-----------------------------------\n')

    print(best_sigma(psi_list, sigmas))
    plot_weight_avg(weighted_average, SAVE_PLOT=SAVE_PLOT)
    for entropy, sigma in zip(entropies, sigmas):
        plot_entropy_acc_cert(entropy, sigma, iters, SAVE_PLOT=SAVE_PLOT)
