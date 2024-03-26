import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import defaultdict


def plot_entropy_acc_cert(ent_acc_cert, labels, sigma, iterations, SAVE_PLOT=False):
    """
    Plots the entropy, accuracy and certainty in a 3D plot.
    """
    entropy, accuracy, certainty = zip(*ent_acc_cert)

    plt.figure(figsize=(15, 9))
    ax = plt.axes(projection='3d')
    im = ax.scatter3D(accuracy, entropy, certainty, c=labels, cmap='tab10')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Entropy')
    ax.set_zlabel('Certainty')
    plt.title(
        f'(Accuracy, Entropy, Certainty) (σ: {sigma}, Iterations: {iterations})')
    plt.grid()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.log(10) + 0.1)
    ax.set_zlim(-0.1, 1.1)
    plt.colorbar(im, ax=ax)

    plot_bounds(classes=10)
    os.makedirs('imgs/entropies', exist_ok=True)
    plt.savefig(
        f'imgs/entropies/entropy_prob_sigma_{sigma:.2f}.pdf') if SAVE_PLOT else plt.show()


def barplot_ent_acc_cert(ent_acc_cert, labels, sigma, SAVE_PLOT=False):
    sums = defaultdict(lambda: np.zeros(3))
    counts = defaultdict(int)

    ent, acc, cert = map(np.array, zip(*ent_acc_cert))
    labels = np.array(labels)

    for label, entropy, accuracy, certainty in zip(labels, ent, acc, cert):
        sums[label] += [entropy, accuracy, certainty]
        counts[label] += 1

    averages = {label: sum / counts[label] for label, sum in sums.items()}
    avg_ent, avg_acc, avg_cert = zip(
        *[averages[label] for label in sorted(averages)])
    avg_acc, avg_cert = np.array(avg_acc) * 100, np.array(avg_cert) * 100
    num_classes = sorted(averages)

    plt.figure(figsize=(15, 9))
    plt.bar(num_classes, avg_acc, color='maroon', width=0.7)
    plt.bar(num_classes, avg_cert, color='blue', width=0.1)
    plt.bar(num_classes, avg_ent, color='green', width=0.3)
    plt.xlabel('Classes')
    plt.ylabel('Percentage')
    plt.title('Average Accuracy, Certainty, and Entropy per Class')
    plt.legend(['Accuracy', 'Certainty', 'Entropy'])
    os.makedirs('imgs/entropies', exist_ok=True)
    plt.savefig(
        f'imgs/entropies/avg_ent_acc_cert_sigma_{sigma:.2f}.pdf') if SAVE_PLOT else plt.show()


def plot_entropy_acc_cert_gif(ent_acc_cert, sigma, iterations, angle_increment=5, elev_increment=1, SAVE_GIF=True):
    """
    Plots the entropy, accuracy and certainty in a 3D plot, rotating around the diagonal.
    """
    entropy, accuracy, certainty = zip(*ent_acc_cert)

    # Setup figure and 3D axis
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(accuracy, entropy, certainty, c=certainty, cmap='viridis')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Entropy')
    ax.set_zlabel('Certainty')
    plt.title(f"""(Accuracy, Entropy, Certainty) (σ: {
              sigma}, Iterations: {iterations})""")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.log(10) + 0.1)
    ax.set_zlim(-0.1, 1.1)

    temp_dir = 'imgs/temp'
    os.makedirs(temp_dir, exist_ok=True)

    filenames = []
    loop = tqdm(range(0, 360, angle_increment),
                desc='Creating frames', leave=False, disable=False)
    for angle in loop:
        elev = 30 + (angle * elev_increment / angle_increment) % 180
        ax.view_init(elev=elev, azim=angle)
        filename = f'{temp_dir}/frame_{angle}.png'
        plt.savefig(filename)
        filenames.append(filename)

    if SAVE_GIF:
        path = 'imgs/gifs'
        os.makedirs(path, exist_ok=True)
        with imageio.get_writer(path, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in filenames:
            os.remove(filename)
        os.rmdir(temp_dir)
    else:
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
    os.makedirs('imgs/curves', exist_ok=True)
    plt.savefig(f'imgs/curves/curves.pdf') if SAVE_PLOT else plt.show()


def visualize_attention_map(image, attention_map):
    attention_map = attention_map.mean(dim=1)
    attention_map_no_class_token = attention_map[:, 1:, 1:]
    attention_map_single = attention_map_no_class_token[0]
    attention_map_avg = attention_map_single.mean(dim=0)

    seq_len = attention_map_avg.shape[0]
    sqrt_len = int(np.sqrt(seq_len))
    attention_map_2d = attention_map_avg.view(sqrt_len, sqrt_len)

    attention_map_resized = torch.nn.functional.interpolate(
        attention_map_2d.unsqueeze(0).unsqueeze(0),
        size=image.shape[-2:],
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    attention_map_np = attention_map_resized.detach().cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(image[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.axis('off')
    plt.title('Original Image')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(image[0].permute(1, 2, 0).detach().cpu().numpy(), alpha=0.8)
    plt.imshow(attention_map_np, cmap='hot',
               interpolation='nearest', alpha=0.3)
    plt.axis('off')
    plt.title('Attention Map')
    plt.colorbar()

    plt.show()
