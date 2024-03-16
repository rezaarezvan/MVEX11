import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_parameters(model):
    original_params = [param.clone() for param in model.parameters()]
    return original_params


@torch.no_grad()
def restore_parameters(model, original_params):
    for param, original in zip(model.parameters(), original_params):
        param.copy_(original)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f'Model loaded from {path}')
    return model


@torch.no_grad()
def add_noise(model, sigma):
    for param in model.parameters():
        param += torch.randn_like(param) * sigma
    return model


def train(model, train_loader, test_loader, optim, epochs=40, lossfn=nn.CrossEntropyLoss()):
    model.to(DEVICE)
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

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), accuracy=accuracy.item())

        print(f"Epoch {epoch+1} Summary - Loss: {np.mean(losses):.4f}, Accuracy: {np.mean(accuracies):.4f}")
        evaluate(model, test_loader)


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    model.to(DEVICE)

    loop = tqdm(test_loader, leave=False)
    accuracies = []
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        accuracies.append(accuracy.item())

    avg_accuracy = np.mean(accuracies)
    print(f"Test Set Accuracy: {avg_accuracy:.4f}")
