import torch
import numpy as np
from torch.distributions import Categorical
from . import SEED

torch.manual_seed(SEED)


def entropy(predictions: torch.Tensor):
    """
    Follows this entropy function:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    log_e
    The function works by first calculating the quantity of the number
    in the list and creates a list where it's frequency is placed in that list.
    For example:
    [1,7,1,7,1]->[0,3,0,0,0,0,0,2]/len([1,7,1,7,1])->[0,(3/5),0,0,0,0,0,(2/5)]
    -------------------------------------------------------------------------
    Perform entropy calculation on every row in the predictions matrix
    """
    rows = predictions.size(0)  # Image
    cols = predictions.size(1)  # Label prediction
    entropy = torch.zeros(rows)
    for i in range(rows):
        probability = torch.bincount(predictions[i], minlength=10).float()
        probability /= cols  # Normalize
        entropy[i] = Categorical(probs=probability).entropy()
    return entropy


def weight_avg(data, window_size=0.25):
    """
    Calculates and prints the weighted average of probabilities for specified windows of entropy.

    :param data: A list of tuples, where each tuple is (entropy, probability)
    :param window_size: The size of the entropy window to calculate averages for
    """
    # Determine the maximum entropy value from the data
    max_entropy = max(data, key=lambda x: x[0])[0]

    start = 0
    result = []

    while start <= max_entropy:
        end = start + window_size

        # Filter data for current window
        filtered_data = [t for t in data if start <= t[0] < end]

        # Continue to next window if no data is found within the current window
        if not filtered_data:
            start += window_size
            continue

        # Add count 1 for each data point in the window (think frequency of the probability)
        weights = len(filtered_data)

        # Calculate the weighted average for the filtered data
        weighted_average_filtered = sum(
            p for (_, p, _) in (filtered_data)) / (weights)

        # Print the weighted average for the current window
        print(
            f"Interval [{start:.3f}, {end:.3f}], Weighted Average: [{weighted_average_filtered:.3f}]")
        result.append(((start+end)/2, weighted_average_filtered))
        start += window_size

    print('-----------------------------------')

    return result


def max_psi_sigma(psi_list, sigma_list):
    """
    Returns max (psi, sigma) based on psi value
    :param psi_list: A list of psi values
    :param sigma_list: A list of sigma values
    """
    max_psi_index = np.argmax(psi_list)
    return psi_list[max_psi_index], sigma_list[max_psi_index % len(sigma_list)]


def pi(alpha, alpha_sigma):
    """
    Computes the "π_σ"-metric defined as
    π_σ = α - α_σ
    where α is the original accuracy and α_σ is the accuracy with noise
    :param alpha: The original accuracy
    :param alpha_sigma: The accuracy with noise
    """
    return alpha - alpha_sigma


def psi(data, _lambda=0.1):
    """
    Computes the "ψ_σ"-metric defined as
    ψ_σ = E[p_σ(H)] - corr_σ(α, H) λ
    :param data: A list of tuples, where each tuple is (entropy, probability)
    :param _lambda: The lambda value to use in the calculation
    """
    ent, pro, _ = zip(*data)
    ent = np.array(ent)
    pro = np.array(pro)

    ent_mean = ent.mean()
    pro_mean = pro.mean()

    # Minimum entropy is ~1e-7 and not 0
    if ent.max() < 1e-6:
        return pro_mean

    ent_norm = ent - ent_mean
    pro_norm = pro - pro_mean

    pairs_norm = ent_norm * pro_norm

    nominator = pairs_norm.mean()

    var_ent = np.var(ent)
    var_pro = np.var(pro)
    denominator = np.sqrt(var_ent * var_pro)

    correlation = nominator / denominator

    return pro_mean - correlation*_lambda
