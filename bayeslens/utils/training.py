import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def save_model(model, path):
    """
    Save model to path
    """
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
