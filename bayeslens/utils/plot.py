import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_prob(ent_prob, sigma, acc, iterations, SAVE_PLOT=False):
    """
    Plots the list of probability/entropy parts on the x/y-axis respectively.
    """
    entropy, probability, certainty = zip(*ent_prob)

    plt.figure(figsize=(15, 9))
    ax = plt.axes(projection='3d')
    ax.scatter3D(probability, entropy, certainty, c=certainty, cmap='viridis')
    plt.xlabel('Probability')
    plt.ylabel('Entropy')
    plt.title(
        f'Entropy to Probability (sigma={sigma}, iterations={iterations}, {acc}%)')
    plt.grid()
    plt.ylim(-0.1, np.log(10) + 0.1)
    plt.xlim(-0.1, 1.1)
    plot_bounds(classes=10)
    if SAVE_PLOT:
        plt.savefig(f'plots/entropies/entropy_prob_sigma_{sigma:.2f}.pdf')
    else:
        plt.show()

    plt.clf()


def plot_bounds(classes):
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


def plot_weight_avg(data, SAVE_PLOT=False):
    """
    Plots the weighted averages of probabilities for specified windows of entropy.

    :param data: A list of tuples, where each tuple is (entropy, probability)
    """
    plt.figure(figsize=(15, 9))
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
    else:
        plt.show()

    plt.clf()
