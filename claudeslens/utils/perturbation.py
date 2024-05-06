import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from collections import defaultdict

from claudeslens.utils import DEVICE, SEED
from claudeslens.utils.metrics import entropy, weight_avg, psi, max_psi_sigma, pi
from claudeslens.utils.plot import plot_entropy_acc_cert, plot_weight_avg, barplot_ent_acc_cert
from claudeslens.models import Pretrained_ViT_B_16, ClaudesLens_ViT, Pretrained_ConvNext, ClaudesLens_ConvNext
from claudeslens.utils.training import evaluate, add_noise, remove_noise, eval_attention, add_noise_attention, remove_noise_attention, eval_features, add_noise_conv_weights, remove_noise_conv_weights

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
        noise_data = add_noise(model, sigma)

        acc = evaluate(model, test_loader)
        pi_list.append(pi(og_acc, acc))

        remove_noise(model, noise_data)
        torch.cuda.empty_cache()

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
            noise = add_noise(model, sigma)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            prob = nn.functional.softmax(outputs, dim=1)
            correct_class_probs = prob.gather(1, labels.unsqueeze(1)).squeeze()
            predictions.append(predicted)
            probs.append(correct_class_probs)

            remove_noise(model, noise)
            torch.cuda.empty_cache()

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
            noise = add_noise(model, sigma)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            prob = nn.functional.softmax(outputs, dim=1)
            correct_class_probs = prob.gather(1, labels.unsqueeze(1)).squeeze()

            predictions.append(preds)
            probs.append(correct_class_probs)

            remove_noise(model, noise)
            torch.cuda.empty_cache()

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


def model_is_uncertain(data, cutoff_acc=0.10):
    '''
    Returns True if the model is uncertain, False otherwise.

    A model is said to be uncertain if a specified confidence interval (default 95%) of the data is below a certain accuracy threshold (default 25%).

    Data is of form EAC_data = [(entropy, accuracy, certainty), ...]
    '''
    data = np.array(data)
    acc = data[:, 1]

    n = len(acc)
    m = np.mean(acc)
    std_err = np.std(acc) / np.sqrt(n)
    z = 1.65  # 90% confidence interval

    ci_low = m - z * std_err
    is_uncertain = ci_low < cutoff_acc

    return is_uncertain


def perturbation(model, test_loader, iters=10, sigmas=[0, 0.01, 0.1, 1], lambdas=[0.1, 0.5, 1], entropy_window_size=0.1,
                 SAVE_PLOT=True, LOAD_DATA=False):
    """
    Main evaluation loop for the perturbation tests for the model
    """

    if LOAD_DATA:
        print("Loading data from file")
        f = open(f"results/{model.__class__.__name__}.json", 'r')
        all_sigma_data = json.load(f)

        for sigma, sigma_data in all_sigma_data["all_sigma_data"].items():
            sigma = float(sigma)
            plot_entropy_acc_cert(sigma_data["ent_acc_cert_weights"], test_loader.dataset.targets, sigma,
                                  iters, sigma_data["weighted_average"], SAVE_PLOT=SAVE_PLOT, type='weight', model_name=model.__class__.__name__)
            barplot_ent_acc_cert(sigma_data["ent_acc_cert_weights"], test_loader.dataset.targets, sigma,
                                 SAVE_PLOT=SAVE_PLOT, type='weight', model_name=model.__class__.__name__)

            temp = weight_avg(
                sigma_data["ent_acc_cert_images"], window_size=entropy_window_size)
            temp = (temp, sigma)
            plot_entropy_acc_cert(sigma_data["ent_acc_cert_images"], test_loader.dataset.targets, sigma,
                                  iters, temp, SAVE_PLOT=SAVE_PLOT, type='image', model_name=model.__class__.__name__)
            barplot_ent_acc_cert(sigma_data["ent_acc_cert_images"], test_loader.dataset.targets, sigma,
                                 SAVE_PLOT=SAVE_PLOT, type='image', model_name=model.__class__.__name__)

            is_uncertain = model_is_uncertain(
                sigma_data["ent_acc_cert_weights"])
            print(
                f"For σ: {sigma}, model is uncertain: {is_uncertain} for weight perturbation")

            is_uncertain = model_is_uncertain(
                sigma_data["ent_acc_cert_images"])
            print(
                f"For σ: {sigma}, model is uncertain: {is_uncertain} for image perturbation")

        # plot_weight_avg(all_sigma_data["weighted_average"], SAVE_PLOT=SAVE_PLOT,
        #                model_name=model.__class__.__name__)

        print("Pair Entanglement")
        print(all_sigma_data["pair_entaglement"])

        return

    weighted_average = []
    list_of_psi_list = []
    pair_entaglement = []
    og_acc = evaluate(model, test_loader)
    print(f"Original Accuracy: {og_acc}")

    all_sigma_data = {}

    for sigma in sigmas:
        sigma_data = {}
        psi_list = []

        pi = evaluate_robustness(model, test_loader, og_acc,
                                 sigma=sigma, iters=iters)
        print(f"σ: {sigma}, π: {pi}")

        print("--------------------------------------------------")
        print("Weight perturbation")
        ent_acc_cert_weights = evaluate_weight_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        print("--------------------------------------------------")
        print("Image perturbation")
        ent_acc_cert_images = evaluate_image_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        weighted_average.append(
            (weight_avg(ent_acc_cert_weights, window_size=entropy_window_size), sigma))

        matrix_with_correct_label = evaluate_pair_entanglement(
            model, test_loader, sigma=sigma, iters=iters, threshold=0)

        pair_entaglement.append(matrix_with_correct_label)

        sigma_data["pi"] = pi
        sigma_data["ent_acc_cert_weights"] = ent_acc_cert_weights
        sigma_data["ent_acc_cert_images"] = ent_acc_cert_images
        sigma_data["weighted_average"] = weighted_average[-1]
        sigma_data["matrix_with_correct_label"] = matrix_with_correct_label

        plot_entropy_acc_cert(ent_acc_cert_weights, test_loader.dataset.targets, sigma,
                              iters, SAVE_PLOT=SAVE_PLOT, type='weight', model_name=model.__class__.__name__)
        barplot_ent_acc_cert(ent_acc_cert_weights, test_loader.dataset.targets, sigma,
                             SAVE_PLOT=SAVE_PLOT, type='weight', model_name=model.__class__.__name__)
        plot_entropy_acc_cert(ent_acc_cert_images, test_loader.dataset.targets, sigma,
                              iters, SAVE_PLOT=SAVE_PLOT, type='image', model_name=model.__class__.__name__)
        barplot_ent_acc_cert(ent_acc_cert_images, test_loader.dataset.targets, sigma,
                             SAVE_PLOT=SAVE_PLOT, type='image', model_name=model.__class__.__name__)

        is_uncertain = model_is_uncertain(ent_acc_cert_weights)
        print(
            f"For σ: {sigma}, model is uncertain: {is_uncertain} for weight perturbation")
        is_uncertain = model_is_uncertain(ent_acc_cert_images)
        print(
            f"For σ: {sigma}, model is uncertain: {is_uncertain} for image perturbation")

        for _lambda in lambdas:
            psi_value = psi(ent_acc_cert_weights, _lambda=_lambda)
            print(f"λ: {_lambda}, ψ: {psi_value}")
            psi_list.append(psi_value)
            sigma_data[f"psi_lambda_{_lambda}"] = psi_value

        all_sigma_data[sigma] = sigma_data
        list_of_psi_list.append(psi_list)
        print('-----------------------------------\n')

    best_psi, best_sigma = max_psi_sigma(list_of_psi_list, sigmas)
    print(f"Max: (ψ: {best_psi}, σ: {best_sigma})")
    plot_weight_avg(weighted_average, SAVE_PLOT=SAVE_PLOT,
                    model_name=model.__class__.__name__)

    print("Pair Entanglement")
    print(pair_entaglement)

    ent_acc_cert_data = {
        "all_sigma_data": all_sigma_data,
        "weighted_average": weighted_average,
        "pair_entaglement": pair_entaglement,
        "model_name": model.__class__.__name__
    }

    os.makedirs('results', exist_ok=True)
    path = f"results/{model.__class__.__name__}.json"

    if os.path.exists(path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)

        data["all_sigma_data"].update(all_sigma_data)

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        with open(path, 'w') as json_file:
            json.dump(ent_acc_cert_data, json_file, indent=4)

    # For attention and feature maps:

    # if isinstance(model, ClaudesLens_ViT) or isinstance(model, Pretrained_ViT_B_16):
    #     for sigma in sigmas:
    #         noise = add_noise_attention(model, sigma)
    #         eval_attention(model, test_loader, n=3,
    #                        sigma=sigma, SAVE_PLOT=SAVE_PLOT)
    #         remove_noise_attention(model, noise)

    # if isinstance(model, ClaudesLens_ConvNext) or isinstance(model, Pretrained_ConvNext):
    #     for sigma in sigmas:
    #         noise = add_noise_conv_weights(model, sigma)
    #         eval_features(model, test_loader, n=3,
    #                       sigma=sigma, SAVE_PLOT=SAVE_PLOT)
    #         remove_noise_conv_weights(model, noise)
