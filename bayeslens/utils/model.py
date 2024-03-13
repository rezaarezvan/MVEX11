import torch
# from .entrop import test_model_noise, calculate_weighted_averages, compute_k, plot_weighted_averages, plot_entropy_prob

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model_parameters(model):
    original_params = [param.clone() for param in model.parameters()]
    return original_params


@torch.no_grad()
def restore_model_parameters(model, original_params):
    for param, original in zip(model.parameters(), original_params):
        param.copy_(original)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print(f'Model loaded from {path}')
    return model


def train_model(model, train, val, test, optimizer, criterion, flatten=False, num_epochs=40):
    model.to(DEVICE)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images) if not flatten else model(
                images.flatten(1))
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train)}], Loss: {loss.item():.4f}')

        test_model(model, test)
        print()


@torch.no_grad()
def test_model(model, test, flatten=False, mode='validation'):
    model.eval()

    correct = 0
    total = 0

    for i, (images, labels) in enumerate(test):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images) if not flatten else model(
            images.flatten(1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i+1) % 10 == 0:
            print(f'Step [{i+1}/{len(test)}]')

    accuracy = 100 * correct / total
    print(f'{mode.capitalize()} accuracy of the model: {accuracy} %')


# def model_with_noise(model, test, sigmas, lambdas, iterations):
#     entropies = []
#     weighted_average = []
#
#     for sigma in sigmas:
#         print(f"Sigma: {sigma}")
#         entropy = test_model_noise(model, test, sigma=sigma, iters=iterations)
#         weighted_average.append((calculate_weighted_averages(entropy), sigma))
#         entropies.append(entropy)
#         for _lambda in lambdas:
#             print(f"Lambda: {_lambda}, K: {
#                   compute_k(entropy, _lambda=_lambda)}")
#         print('-----------------------------------\n')
#
#     plot_weighted_averages(weighted_average)
#     for entropy, sigma in zip(entropies, sigmas):
#         plot_entropy_prob(entropy, sigma, 10,
#                           iterations)
