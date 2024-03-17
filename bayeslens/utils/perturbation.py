import torch
from tqdm.auto import tqdm
from .metrics import entropy, weight_avg, psi, best_sigma
from .training import add_noise, restore_parameters, save_parameters, evaluate
from .plot import plot_entropy_prob, plot_weight_avg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)


@torch.no_grad()
def evalute_robustness(model, test_loader, sigmas=[0.1, 0.25, 0.5], iters=10):
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
def evalute_perturbation(model, test_loader, sigma=0, iters=10):
    model.eval()
    model.to(DEVICE)
    result = []

    loop = tqdm(test_loader, leave=False)
    for images, labels in loop:
        predictions = []
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        for _ in range(iters):
            original_params = save_parameters(model)
            add_noise(model, sigma)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            predictions.append(preds)
            restore_parameters(model, original_params)

        pred_matrix = torch.stack(predictions, dim=1)
        ent = entropy(pred_matrix)
        accuracy = (pred_matrix == labels.view(-1, 1)).float().mean(dim=1)

        for e, a in zip(ent, accuracy):
            result.append((e.item(), a.item()))

    return result


def perturbation(model, test_loader, iters=10, sigmas=[0, 0.25, 0.5], lambdas=[0.1, 0.5, 1], entrop_window_size=0.1):
    entropies = []
    weighted_average = []
    psi_list = []

    for sigma in sigmas:
        print(f"σ: {sigma}")
        entropy = evalute_perturbation(
            model, test_loader, sigma=sigma, iters=iters)
        weighted_average.append(
            (weight_avg(entropy, window_size=entrop_window_size), sigma))
        entropies.append(entropy)
        for _lambda in lambdas:
            print(f"λ: {_lambda}, ψ: {psi(entropy, _lambda=_lambda)}")
        psi_list.append(psi(entropy))
        print('-----------------------------------\n')

    print(best_sigma(psi_list, sigmas))
    plot_weight_avg(weighted_average, )
    for entropy, sigma in zip(entropies, sigmas):
        plot_entropy_prob(entropy, sigma, 0, iters)
