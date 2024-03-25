import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm



def plot_most_often_similar(predictions_and_correct_label, threshold, labels):
    """
    Plots the most often interchanged predictions.
    Args:
        labels: Integer amount of labels to compare
    """
    # Takes out the lists where |labels| amount of classes account for more
    # than |threshold| amount of predictions, for example:
    # (([1,2,1,2,1,2,1,3] 1), 0.8, 2)
    # 1 and 2 account for more than 80% of the contents of the list
    # and were mistaken for each other.

    # Remove prediction instance if one label is  80% majority
    frequency = []
    remove_majority = 0.8*len(predictions_and_correct_label[0][0])

    # Could use some testing
    for instance, label in predictions_and_correct_label:
        occurrence_list = torch.bincount(instance)
        add_to_frequency = True
        for occurrence in occurrence_list:
            if occurrence >= remove_majority:
                add_to_frequency = False
                continue
        if add_to_frequency:
            frequency.append(occurrence_list)

    print(frequency)
    print("newprint")
    pairs_of_interest = []
    elements_of_interest = []
    for prediction in frequency:
        add_to_interest = True
        for element1 in prediction:
            for element2 in prediction:
                if element1 + element2 > threshold*remove_majority:
                    pairs_of_interest.append(prediction)

    print(pairs_of_interest)
    return None




def plot_entropy_acc_cert(ent_acc_cert, sigma, iterations, SAVE_PLOT=False):
    """
    Plots the entropy, accuracy and certainty in a 3D plot.
    """
    entropy, accuracy, certainty = zip(*ent_acc_cert)

    plt.figure(figsize=(15, 9))
    ax = plt.axes(projection='3d')
    ax.scatter3D(accuracy, entropy, certainty, c=certainty, cmap='viridis')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Entropy')
    ax.set_zlabel('Certainty')
    plt.title(
        f'(Accuracy, Entropy, Certainty) (σ: {sigma}, Iterations: {iterations})')
    plt.grid()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.log(10) + 0.1)
    ax.set_zlim(-0.1, 1.1)

    plot_bounds(classes=10)
    plt.savefig(
        f'plots/entropies/entropy_prob_sigma_{sigma:.2f}.pdf') if SAVE_PLOT else plt.show()


def plot_entropy_acc_cert_gif(ent_acc_cert, sigma, iterations, angle_increment=5, elev_increment=1, SAVE_GIF=True, gif_path='entropy_acc_cert_diagonal.gif'):
    """
    Plots the entropy, accuracy and certainty in a 3D plot, rotating around the diagonal.
    """
    entropy, accuracy, certainty = zip(*ent_acc_cert)

    # Setup figure and 3D axis
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    img = ax.scatter3D(accuracy, entropy, certainty,
                       c=certainty, cmap='viridis')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Entropy')
    ax.set_zlabel('Certainty')
    plt.title(f"""(Accuracy, Entropy, Certainty) (σ: {
              sigma}, Iterations: {iterations})""")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.log(10) + 0.1)
    ax.set_zlim(-0.1, 1.1)

    # Directory for temporary image files
    temp_dir = 'temp_images'
    os.makedirs(temp_dir, exist_ok=True)

    # Create frames
    filenames = []
    loop = tqdm(range(0, 360, angle_increment),
                desc='Creating frames', leave=False, disable=False)
    for angle in loop:
        # Adjust elevation based on angle
        elev = 30 + (angle * elev_increment / angle_increment) % 180
        ax.view_init(elev=elev, azim=angle)
        filename = f'{temp_dir}/frame_{angle}.png'
        plt.savefig(filename)
        filenames.append(filename)

    # Create GIF
    if SAVE_GIF:
        with imageio.get_writer(gif_path, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Optional: Cleanup
        for filename in filenames:
            os.remove(filename)
        os.rmdir(temp_dir)
    else:
        # If not saving, just display the last frame
        plt.show()

    plt.close(fig)


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
    plt.savefig(f'plots/curves/curves.pdf') if SAVE_PLOT else plt.show()
