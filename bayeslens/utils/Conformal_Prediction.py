import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import math

def load_data(calibration_points=2500, batch_size=32):

    train = torchvision.datasets.MNIST(
        root='../../extra/datasets', train=True, download=True, transform=transforms.ToTensor())
    test = torchvision.datasets.MNIST(
        root='../../extra/datasets', train=False, download=True, transform=transforms.ToTensor())


    calibration_indices = list(range(calibration_points))
    test_indices = list(range(calibration_points, len(test)))


    calibration = Subset(test, calibration_indices)
    test = Subset(test, test_indices)

    train = DataLoader(train, batch_size=32, shuffle=True)
    test = DataLoader(test, batch_size=32, shuffle=False)
    calibration = DataLoader(calibration, batch_size=32, shuffle=False)

    return train, test, calibration

@torch.no_grad()
def calculate_q_hat(model, calibration, alpha, device):
    model.eval()
    E = []
    for image, label in calibration:
        image = image.to(device)
        label = label.to(device)
        output = model(image.flatten(1))
        sorted_scores, sorted_indices = F.softmax(output, dim=1).sort(dim=1, descending=True)
        
        # Calculate the calibration score for each image and append to E
        for idx, true_label in enumerate(label):
            true_label_rank = (sorted_indices[idx] == true_label).nonzero(as_tuple=True)[0].item()
            calibration_score = sorted_scores[idx, :true_label_rank+1].sum().item()
            E.append(1-calibration_score)

    n = 2500
    #q_hat = np.quantile(E, alpha, method='lower')
    #q_hat = np.quantile(E, (1-alpha), method='higher')
    q_hat = np.quantile(E, math.ceil((n+1)*(1-alpha))/n, method='higher')

    return q_hat

def plot_image(image, prediction_set, label):
    plt.imshow(image.cpu()[0], cmap='gray')
    plt.title(f"Prediction Set: {prediction_set}\nTrue Label: {label}")
    plt.show()

def plot_images_with_predictions(batch_images, prediction_mask, class_names=None):
    """
    Plot a batch of MNIST images with prediction masks as titles.
    
    Parameters:
    - batch_images: Tensor of shape (B, C, H, W) where B is the batch size, C is the number of channels (1 for MNIST),
      H is the height, and W is the width of the images.
    - prediction_mask: Boolean Tensor of shape (B, N) where B is the batch size and N is the number of classes.
      True values indicate the class is part of the prediction set for the image.
    - class_names: Optional list of class names. If not provided, class indices are used as names.
    """
    # Adjusting the size of the plot depending on the batch size for better visibility
    num_images = len(batch_images)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2.5, 2.5))
    if num_images == 1:  # If there's only one image, axs is not an array
        axs = [axs]
    
    for img, mask, ax in zip(batch_images, prediction_mask, axs):
        img = img.squeeze()  # Remove channel dim since it's 1 for MNIST, resulting in H x W
        ax.imshow(img.cpu().numpy(), cmap='gray')
        ax.axis('off')
        
        # Generate title based on the prediction mask
        if class_names:
            titles = [class_names[i] for i, include in enumerate(mask) if include]
        else:
            titles = [str(i) for i, include in enumerate(mask) if include]
        ax.set_title(", ".join(titles))
    
    plt.tight_layout()
    plt.show()

class_names = [str(i) for i in range(10)]

def ConformalPrediction(model,calibration, test, alpha, device):
    model.eval()
    q_hat = calculate_q_hat(model, calibration, alpha, device)
    predictions = []
    coverage_counter = 0
    counter = 0
    for images, labels in test:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images.flatten(1))
        output, _ = F.softmax(output, dim=1).sort(dim=1, descending=True)

        for idx, label in enumerate(labels):
            cumsum_probs = output[idx].cumsum(0)
            prediction_set = (output[idx] >= (1-q_hat)).nonzero(as_tuple=True)[0].tolist()
            #plot_image(images[idx], prediction_set, label)
            if label.item() in prediction_set:
                coverage_counter += 1
            counter += 1
    coverage = coverage_counter / counter
    print(f"Empirical Coverage: {coverage*100:.2f}%")
    exit()
        #cumsum_probs = output.cumsum(dim=1)
        #prediction_mask = cumsum_probs <= q_hat
        #
        #predictions.append(prediction_mask)
        #plot_images_with_predictions(images[:5], prediction_mask[:5], class_names)

        ## if the prediction_mask has included correct label then add 1 to included
        #included += (prediction_mask )
        
        #cumsum_probs = output[0].cumsum(0)
        #prediction_set = (cumsum_probs <= (q_hat)).nonzero(as_tuple=True)[0].tolist()

        ##prediction_set = [x for x in output[0] if x.item() > q_hat]
        #plot_image(images[0].cpu().numpy().squeeze(), prediction_set)

        #print(output[0].shape)
        #print("Tested conformal pred")
        #exit()

    return predictions
