import torch

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


