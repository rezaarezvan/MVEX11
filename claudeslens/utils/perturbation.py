import torch
import torch.nn as nn
from collections import defaultdict
from tqdm.auto import tqdm
from claudeslens.utils.metrics import entropy, weight_avg, psi, max_psi_sigma, pi
from claudeslens.utils.training import add_noise, restore_parameters, save_parameters, evaluate
from claudeslens.utils.plot import plot_entropy_acc_cert, plot_weight_avg, barplot_ent_acc_cert
from . import DEVICE, SEED

torch.manual_seed(SEED)

"""
Everything calculated in this file is done batch-wise.
"""


@torch.no_grad()
def evaluate_robustness(model, test_loader, og_acc, sigma=0, iters=10):
    """
    Evaluate the robustness of the model by adding noise to the model parameters
    and evaluating the models accuracy on the entire test set
    """
    model.eval().to(DEVICE)
    pi_list = []
    for _ in range(iters):
        original_params = save_parameters(model)
        add_noise(model, sigma)

        acc = evaluate(model, test_loader)
        pi_list.append(pi(og_acc, acc))

        restore_parameters(model, original_params)

    pi_mean = sum(pi_list) / len(pi_list)
    return pi_mean


@torch.no_grad()
def evaluate_pair_entanglement(model, test_loader, sigma, threshold, iters):
    """
    Returns:
        A list of tuples with the structure: (index of largest, index of second largest, frequency, sigma)
    """
    model.eval().to(DEVICE)

    loop = tqdm(test_loader, leave=False)
    for images, labels in loop:
        predictions = []
        probs = []
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        for _ in range(iters):
            original_params = save_parameters(model)
            add_noise(model, sigma)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            prob = nn.functional.softmax(outputs, dim=1)
            correct_class_probs = prob.gather(1, labels.unsqueeze(1)).squeeze()
            predictions.append(predicted)
            probs.append(correct_class_probs)

            restore_parameters(model, original_params)

        all_predictions = torch.stack(predictions, dim=1)

    # List comprehension adding together each element into their specific row
    # This will sort the list as following:
    # [((predictions),correct label) , ...]
    #
    matrix_with_correct_label = [
        (all_predictions[i], labels[i].item()) for i in range(len(labels))]

    # If to plot the entanglement, use the matrix returned below
    # return matrix_with_correct_label
    #
    frequency = []
    # remove_majority = threshold * len(matrix_with_correct_label[0][0])
    for instance, label in matrix_with_correct_label:
        occurrence_list = torch.bincount(instance)
        # if all(occurrence < remove_majority for occurrence in occurrence_list):
        frequency.append((occurrence_list, label))

    pairs_of_interest_with_label = []
    for bincount_list, label in frequency:
        sorted_indices = torch.argsort(bincount_list, descending=True)
        if len(sorted_indices) > 1:
            index_of_largest = sorted_indices[0].item()
            index_second_largest = sorted_indices[1].item()
            pairs_of_interest_with_label.append(
                (index_of_largest, index_second_largest, label))

    frequency_map = defaultdict(int)
    for largest, second_largest, _ in pairs_of_interest_with_label:
        key = (largest, second_largest)
        frequency_map[key] += 1

    entanglement_list = [
        (largest, second_largest, count, sigma)
        for (largest, second_largest), count in frequency_map.items()
    ]

    return entanglement_list


@torch.no_grad()
def evaluate_weight_perturbation(model, test_loader, sigma=0, iters=10):
    """
    Evaluate the perturbation of the model by adding noise to the model parameters
    and calculating the entropy, accuracy and certainty of the model for each batch
    """
    model.eval().to(DEVICE)
    result = []

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


@torch.no_grad()
def evaluate_image_perturbation(model, test_loader, sigma=0, iters=10):
    """
    Evaluate the perturbation of the model by adding noise to the input images
    and calculating the entropy, accuracy and certainty of the model for each batch
    """
    model.eval().to(DEVICE)
    result = []

    loop = tqdm(test_loader, leave=False)
    for images, labels in loop:
        predictions = []
        probs = []
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        original_images = images.clone()
        for _ in range(iters):
            images = original_images + torch.randn_like(images) * sigma
            images = torch.clamp(images, 0, 1)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            prob = nn.functional.softmax(outputs, dim=1)
            correct_class_probs = prob.gather(1, labels.unsqueeze(1)).squeeze()

            predictions.append(preds)
            probs.append(correct_class_probs)

        pred_matrix = torch.stack(predictions, dim=1)
        prob = torch.stack(probs).mean(dim=0)
        ent = entropy(pred_matrix)
        accuracy = (pred_matrix == labels.view(-1, 1)).float().mean(dim=1)

        for e, a, c in zip(ent, accuracy, prob):
            result.append((e.item(), a.item(), c.item()))

    return result


def perturbation(model, test_loader, iters=20, sigmas=[0, 0.01, 0.1, 1], lambdas=[0.1, 0.5, 1], entropy_window_size=0.1,
                 SAVE_PLOT=True):
    """
    Main evaluation loop for the perturbation tests for the model
    """
    EAC_weights = []
    EAC_images = []
    weighted_average = []
    psi_list = []
    pair_entaglement = []
    og_acc = evaluate(model, test_loader)

    for sigma in sigmas:
        pi = evaluate_robustness(model, test_loader, og_acc,
                                 sigma=sigma, iters=iters)
        print(f"σ: {sigma}, π: {pi}")
        ent_acc_cert_weights = evaluate_weight_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        ent_acc_cert_images = evaluate_image_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        weighted_average.append(
            (weight_avg(ent_acc_cert_weights, window_size=entropy_window_size), sigma))
        EAC_weights.append(ent_acc_cert_weights)
        EAC_images.append(ent_acc_cert_images)

        matrix_with_correct_label = evaluate_pair_entanglement(
            model, test_loader, sigma=sigma, iters=iters, threshold=0)

        pair_entaglement.append(matrix_with_correct_label)

        for _lambda in lambdas:
            print(
                f"λ: {_lambda}, ψ: {psi(ent_acc_cert_weights, _lambda=_lambda)}")
            psi_list.append(psi(ent_acc_cert_weights, _lambda=_lambda))
        print('-----------------------------------\n')

    best_psi, best_sigma = max_psi_sigma(psi_list, sigmas)
    print(f"Max: (ψ: {best_psi}, σ: {best_sigma})")
    plot_weight_avg(weighted_average, SAVE_PLOT=SAVE_PLOT, model_name=model.__class__.__name__)

    print("Weight Perturbation")
    for ent_acc_cert_weights, sigma in zip(EAC_weights, sigmas):
        plot_entropy_acc_cert(ent_acc_cert_weights, test_loader.dataset.targets, sigma,
                              iters, SAVE_PLOT=SAVE_PLOT, type='weight', model_name=model.__class__.__name__)
        barplot_ent_acc_cert(ent_acc_cert_weights, test_loader.dataset.targets, sigma,
                             SAVE_PLOT=SAVE_PLOT, type='weight', model_name=model.__class__.__name__)

    print("Image Perturbation")
    for ent_acc_cert_images, sigma in zip(EAC_images, sigmas):
        plot_entropy_acc_cert(ent_acc_cert_images, test_loader.dataset.targets, sigma,
                              iters, SAVE_PLOT=SAVE_PLOT, type='image', model_name=model.__class__.__name__)
        barplot_ent_acc_cert(ent_acc_cert_images, test_loader.dataset.targets, sigma,
                             SAVE_PLOT=SAVE_PLOT, type='image', model_name=model.__class__.__name__)
    print("Pair Entanglement")
    print(pair_entaglement)

    '''
    For attention and feature maps:
    if isinstance(model, ClaudesLens_ViT) or isinstance(model, Pretrained_ViT_B_16):
        eval_attention(model, test_loader)
    '''
