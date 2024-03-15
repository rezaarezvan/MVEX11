import torch
from .metrics import entropy
from .helpers import add_noise, restore_model_parameters, save_model_parameters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)


@torch.no_grad()
def test_model_noise(model, dataset, sigma=0, iters=10, CNN=True):
    model.eval()
    result = []

    for i, (data, target) in enumerate(dataset):
        predictions = []
        data, target = data.to(DEVICE), target.to(DEVICE)
        for _ in range(iters):
            original_params = save_model_parameters(model)
            add_noise(model, sigma)

            data = data if CNN else data.flatten(start_dim=1)
            output = model(data)
            pred = torch.softmax(output, dim=1).argmax(dim=1)

            predictions.append(pred)
            restore_model_parameters(model, original_params)

        pred_matrix = torch.stack(predictions, dim=1)
        ent = entropy(pred_matrix)
        accuracy = (pred_matrix == target.view(-1, 1)).float().mean(dim=1)

        for e, a in zip(ent, accuracy):
            result.append((e.item(), a.item()))

        if (i + 1) % 50 == 0:
            print(f'Step [{i + 1}/{len(dataset)}]')
    print('-----------------------------------')

    return result
