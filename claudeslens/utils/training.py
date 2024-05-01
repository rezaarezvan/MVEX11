import os
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

from claudeslens.models.vit_b_16 import CustomEncoderBlock
from claudeslens.utils import DEVICE, SEED
from claudeslens.utils.plot import visualize_attention_map, visualize_feature_maps

torch.manual_seed(SEED)


@torch.no_grad()
def add_noise(model, sigma):
    noise_data = []
    for param in model.parameters():
        noise = torch.randn_like(param) * sigma
        param.add_(noise)
        noise_data.append(noise)
    return noise_data


@torch.no_grad()
def remove_noise(model, noise_data):
    for param, noise in zip(model.parameters(), noise_data):
        param.sub_(noise)


@torch.no_grad()
def add_noise_attention(model, sigma):
    """
    Add noise to the model attention weights
    """
    model.eval().to(DEVICE)
    noise_data = []
    for module in model.modules():
        if isinstance(module, CustomEncoderBlock):
            attention = module.encoder_block.self_attention
            for param_name, param in attention.named_parameters():
                if 'weight' in param_name:
                    noise = torch.randn_like(param) * sigma
                    param.add_(noise)
                    noise_data.append(noise)
    return noise_data


def remove_noise_attention(model, noise_data):
    """
    Remove noise from the model attention weights
    """
    model.eval().to(DEVICE)
    noise_iter = iter(noise_data)
    for module in model.modules():
        if isinstance(module, CustomEncoderBlock):
            attention = module.encoder_block.self_attention
            for param_name, param in attention.named_parameters():
                if 'weight' in param_name:
                    noise = next(noise_iter, None)
                    if noise is not None:
                        param.sub_(noise)
                    else:
                        raise RuntimeError(
                            "Mismatch in the number of attention weights and saved noise.")


@torch.no_grad()
def add_noise_conv_weights(model, sigma):
    noise_data = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            noise = torch.randn_like(module.weight) * sigma
            module.weight.add_(noise)
            noise_data.append(noise)
            if module.bias is not None:
                noise = torch.randn_like(module.bias) * sigma
                module.bias.add_(noise)
                noise_data.append(noise)
    return noise_data


def remove_noise_conv_weights(model, noise_data):
    noise_iter = iter(noise_data)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            noise = next(noise_iter, None)
            if noise is not None:
                module.weight.sub_(noise)
                if module.bias is not None:
                    noise = next(noise_iter, None)
                    module.bias.sub_(noise)
            else:
                raise RuntimeError(
                    "Mismatch in the number of Conv2d layers and saved noise.")


def save_model(model, path):
    """
    Save model to path, if the path does not exist, it will be created
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model, path):
    """
    Load model from path
    """
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f'Model loaded from {path}')
    return model


def train(model, train_loader, test_loader, optim, epochs=40, lossfn=nn.CrossEntropyLoss(), writer=None):
    """
    Train the model on the given train- and evaluate on test-loaders
    """
    model.to(DEVICE)
    global_step = 0
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        losses, accuracies = [], []
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            out = model(images)
            loss = lossfn(out, labels)

            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            preds = out.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

            losses.append(loss.item())
            accuracies.append(accuracy.item())

            if writer:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Accuracy/train',
                                  accuracy.item(), global_step)

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), accuracy=accuracy.item())

            global_step += 1

        evaluate(model, test_loader, writer, global_step)

    if writer:
        writer.close()


@torch.no_grad()
def evaluate(model, test_loader, writer=None, global_step=None):
    """
    Evaluate given model on the test set
    """
    model.eval().to(DEVICE)
    loop = tqdm(test_loader, leave=False)
    accuracies = []
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        accuracies.append(accuracy.item())

    avg_accuracy = np.mean(accuracies)
    if writer and global_step is not None:
        writer.add_scalar('Accuracy/test', avg_accuracy, global_step)

    print(f"Validation Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy


def eval_attention(model, test_loader, n=3, sigma=0, SAVE_PLOT=False):
    model.eval().to(DEVICE)
    images = next(iter(test_loader))[0].to(DEVICE)
    for idx, image in enumerate(images[:n]):
        image = image.unsqueeze(0)
        _, attention_map = model(image, need_weights=True)
        visualize_attention_map(
            image, attention_map, sigma, SAVE_PLOT, model.__class__.__name__, idx)


def eval_features(model, test_loader, n=3, sigma=0, SAVE_PLOT=False):
    model.eval().to(DEVICE)
    images = next(iter(test_loader))[0].to(DEVICE)
    for idx, image in enumerate(images[:n]):
        image = image.unsqueeze(0)
        model(image, return_features=True)
        visualize_feature_maps(model.feature_maps, sigma,
                               SAVE_PLOT, model.__class__.__name__, idx)
