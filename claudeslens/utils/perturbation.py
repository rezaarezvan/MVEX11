import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .metrics import entropy, weight_avg, psi, max_psi_sigma
from .training import add_noise, restore_parameters, save_parameters, evaluate
from .plot import plot_entropy_acc_cert, plot_weight_avg, barplot_ent_acc_cert
from . import DEVICE, SEED

torch.manual_seed(SEED)


@torch.no_grad()
def evaluate_robustness(model, test_loader, sigmas=[0.1, 0.25, 0.5], iters=10):
    """
    Evaluate the robustness of the model by adding noise to the model parameters
    and evaluating the models accuracy on the entire test set
    """
    model.eval().to(DEVICE)
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
def evalute_weight_perturbation(model, test_loader, sigma=0, iters=10):
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
def evalute_image_perturbation(model, test_loader, sigma=0, iters=10):
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


def perturbation(model, test_loader, iters=100, sigmas=[0.1, 0.2, 0.3], lambdas=[0.1, 0.5, 1], entrop_window_size=0.1, SAVE_PLOT=False):
    """
    Main evaluation loop for the pertubation tests for the model
    """
    EAC_weights = []
    EAC_images = []
    weighted_average = []
    psi_list = []

    for sigma in sigmas:
        print(f"σ: {sigma}")
        ent_acc_cert_weights = evalute_weight_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        ent_acc_cert_images = evalute_image_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        weighted_average.append(
            (weight_avg(ent_acc_cert_weights, window_size=entrop_window_size), sigma))
        EAC_weights.append(ent_acc_cert_weights)
        EAC_images.append(ent_acc_cert_images)
        for _lambda in lambdas:
            print(
                f"λ: {_lambda}, ψ: {psi(ent_acc_cert_weights, _lambda=_lambda)}")
            psi_list.append(psi(ent_acc_cert_weights, _lambda=_lambda))
        print('-----------------------------------\n')

    best_psi, best_sigma = max_psi_sigma(psi_list, sigmas)
    print(f"Max: (ψ: {best_psi}, σ: {best_sigma})")
    plot_weight_avg(weighted_average, SAVE_PLOT=SAVE_PLOT)

    for ent_acc_cert_weights, sigma in zip(EAC_weights, sigmas):
        plot_entropy_acc_cert(ent_acc_cert_weights, test_loader.dataset.targets, sigma,
                              iters, SAVE_PLOT=SAVE_PLOT)
        barplot_ent_acc_cert(ent_acc_cert_weights, test_loader.dataset.targets, sigma,
                             SAVE_PLOT=SAVE_PLOT)

    print("Image Perturbation")
    for ent_acc_cert_images, sigma in zip(EAC_images, sigmas):
        plot_entropy_acc_cert(ent_acc_cert_images, test_loader.dataset.targets, sigma,
                              iters, SAVE_PLOT=SAVE_PLOT)
        barplot_ent_acc_cert(ent_acc_cert_images, test_loader.dataset.targets, sigma,
                             SAVE_PLOT=SAVE_PLOT)
