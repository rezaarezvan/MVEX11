import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


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
