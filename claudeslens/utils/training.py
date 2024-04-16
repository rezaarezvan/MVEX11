import os
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

from claudeslens.models.vit_b_16 import CustomEncoderBlock
from claudeslens.utils import DEVICE, SEED
from claudeslens.utils.plot import visualize_attention_map, visualize_feature_maps

torch.manual_seed(SEED)


def save_parameters(model):
    """
    Save the model parameters
    """
    original_params = [param.clone() for param in model.parameters()]
    return original_params


@torch.no_grad()
def restore_parameters(model, original_params):
    """
    Restore the model parameters to the original values
    """
    for param, original in zip(model.parameters(), original_params):
        param.copy_(original)


def save_attention(model):
    attention_weights = []
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            weights = {
                'in_proj_weight': module.in_proj_weight.clone(),
                'in_proj_bias': module.in_proj_bias.clone(),
                'out_proj_weight': module.out_proj.weight.clone(),
                'out_proj_bias': module.out_proj.bias.clone()
            }
            attention_weights.append(weights)
    return attention_weights


def restore_attention(model, attention_weights):
    weight_iter = iter(attention_weights)
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            weights = next(weight_iter, None)
            if weights is not None:
                module.in_proj_weight.copy_(weights['in_proj_weight'])
                module.in_proj_bias.copy_(weights['in_proj_bias'])
                module.out_proj.weight.copy_(weights['out_proj_weight'])
                module.out_proj.bias.copy_(weights['out_proj_bias'])
            else:
                raise RuntimeError(
                    "Mismatch in the number of MultiheadAttention modules and saved weights.")


def save_conv_weights(model):
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weights.append({
                'weight': module.weight.clone(),
                'bias': module.bias.clone() if module.bias is not None else None
            })
    return weights


def restore_conv_weights(model, saved_weights):
    weight_iter = iter(saved_weights)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            current_weights = next(weight_iter, None)
            if current_weights is not None:
                module.weight.copy_(current_weights['weight'])
                if current_weights['bias'] is not None:
                    module.bias.copy_(current_weights['bias'])
            else:
                raise RuntimeError(
                    "Mismatch in the number of Conv2d layers and saved weights.")


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


@torch.no_grad()
def add_noise(model, sigma):
    """
    Add noise to the model parameters
    """
    for param in model.parameters():
        param += torch.randn_like(param) * sigma
    return model


@torch.no_grad()
def add_noise_attention(model, sigma):
    """
    Add noise to the model attention weights
    """
    for module in model.modules():
        if isinstance(module, CustomEncoderBlock):
            attention = module.encoder_block.self_attention

            for param_name, param in attention.named_parameters():
                if 'weight' in param_name:
                    param += torch.randn_like(param) * sigma


@torch.no_grad()
def add_noise_conv_weights(model, sigma):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight += torch.randn_like(module.weight) * sigma
            if module.bias is not None:
                module.bias += torch.randn_like(module.bias) * sigma


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


def eval_attention(model, test_loader, n=3):
    model.eval().to(DEVICE)
    images = next(iter(test_loader))[0].to(DEVICE)
    for image in images[:n]:
        image = image.unsqueeze(0)
        _, attention_map = model(image, need_weights=True)
        visualize_attention_map(image, attention_map)


def eval_features(model, test_loader, n=3):
    model.eval().to(DEVICE)
    images = next(iter(test_loader))[0].to(DEVICE)
    model.saved_feature_maps = {}

    for image in images[:n]:
        image = image.unsqueeze(0)
        model(image)

    if 'last_conv_output' in model.saved_feature_maps:
        all_feature_maps = torch.cat(
            model.saved_feature_maps['last_conv_output'], dim=0)
        visualize_feature_maps(
            images[:len(all_feature_maps)], all_feature_maps)
    else:
        print("No feature maps saved. Check the hook and model configuration.")
