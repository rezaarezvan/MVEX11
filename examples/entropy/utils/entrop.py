from torch.distributions import Categorical 
import matplotlib.pyplot as plt
import numpy as np
import torch
from .model import save_model_parameters, restore_model_parameters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

@torch.no_grad()
def add_noise(model, sigma):
    for param in model.parameters():
        param += torch.randn_like(param) * sigma
    return model

def calculate_entropy(predictions: torch.Tensor):
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
    entropy = torch.zeros(predictions.size(0))
    for i in range(predictions.size(0)):
        probability  = torch.bincount(predictions[i], minlength=10).float()
        probability /= predictions.size(1) # Normalize
        entropy[i]   = Categorical(probs=probability).entropy()
    return entropy

@torch.no_grad()
def test_model_noise(model, dataset, sigma=0, iters=10, CNN=False):
    model.eval()
    result = []

    print(f"Testing model with noise scale {sigma:.2f}...")
    for batch, (data, target) in enumerate(dataset):
        predictions = []
        data, target = data.to(DEVICE), target.to(DEVICE)
        for _ in range(iters):
            original_params = save_model_parameters(model)
            add_noise(model, sigma)

            data   = data.flatten(1) if not CNN else data
            output = model(data)
            pred   = torch.softmax(output, dim=1).argmax(dim=1)

            predictions.append(pred)
            restore_model_parameters(model, original_params)
        
        pred_matrix = torch.stack(predictions, dim=1)
        entropy     = calculate_entropy(pred_matrix)
        accuracy    = (pred_matrix == target.view(-1, 1)).float().mean(dim=1)
        
        for e, a in zip(entropy, accuracy):
            result.append((e.item(), a.item()))

        if (batch + 1) % 50 == 0:
            print(f'Batch {batch + 1}/{len(dataset)} completed')

    print(f'Batch {batch+1}/{len(dataset)} completed')
    print(f'Testing model with noise scale {sigma:.2f} completed\n')
    return result

def plot_entropy_prob(ent_prob, sigma, acc, iterations, SAVE_PLOT=False):
    """
    Plots the list of probability/entropy parts on the x/y-axis respectively.
    """
    entropy, probability = zip(*ent_prob)

    plt.figure(figsize=(10, 6))
    plt.scatter(probability, entropy, alpha=0.1)
    plt.xlabel('Probability')
    plt.ylabel('Entropy')
    plt.title(f'Entropy to Probability (sigma={sigma}, iterations={iterations}, {acc}%)')
    plt.grid()
    plt.ylim(-0.1, np.log(10) + 0.1)
    plt.xlim(-0.1, 1.1)
    plot_curve(classes=10)
    if SAVE_PLOT: plt.savefig(f'plots/entropies/entropy_prob_sigma_{sigma:.2f}.pdf')
    else: plt.show()


def plot_curve(classes):
    """
    This call plots the domain of the amount of given classes.
    (Beware there exists off-by-one, this is handled in the function)
    """
    classes -= 1
    # Arc-function at bottom
    x = np.linspace(0.0001, 0.9999, 10000)
    y = -x * np.log(x) - (1 - x) * np.log(1 - x)
    plt.plot(x, y, color='orange')
    x = np.linspace(0.0001, 0.9999, 10000)
    y = -x * np.log(x) - (1 - x) * np.log(1 - x)

    # Higher top function
    y2 = []
    for p in x:
        a = -p * np.log(p)
        for _ in range(classes):
            a -= (1 - p) / (classes) * np.log((1 - p) / classes)
        y2.append(a)

    plt.plot(x, y, color="orange")
    plt.plot(x, y2, color="orange")

    # This plots the vertical line
    plt.plot([0, 0], [0, y2[0]], color="orange")

def calculate_weighted_averages(data, window_size=0.25):
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
        #weights = [1 for _ in filtered_data]  
        weights = len(filtered_data)
        
        # Calculate the weighted average for the filtered data
        weighted_average_filtered = sum(p for (_, p) in (filtered_data)) / (weights)
        
        # Print the weighted average for the current window
        print(f"Window {start:.2f} to {end:.2f}: Weighted Average = {weighted_average_filtered}")
        result.append(((start+end)/2, weighted_average_filtered)) 
        start += window_size
    return result

def plot_weighted_averages(data, SAVE_PLOT=False):
    """
    Plots the weighted averages of probabilities for specified windows of entropy.
    
    :param data: A list of tuples, where each tuple is (entropy, probability)
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel('Entropy')
    plt.ylabel('Weighted Average Probability')
    plt.title('Weighted Averages of Probabilities for Specified Windows of Entropy')
    plt.grid()

    for lst, sigma in data:
        sorted_data = sorted(lst, key=lambda x: x[0])
        entropy, probability = zip(*sorted_data)
        plt.plot(entropy, probability, marker='o', label=f'Sigma:{sigma:.2f}')

    plt.legend()
    if SAVE_PLOT:
        plt.savefig(f'plots/curves/curves.pdf')
    else: plt.show()








