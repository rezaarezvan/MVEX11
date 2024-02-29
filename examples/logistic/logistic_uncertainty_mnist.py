import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

torch.manual_seed(0)


def load_data():
    trainset = torchvision.datasets.MNIST(
        root='../../extra/datasets', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(
        root='../../extra/datasets', train=False, download=True, transform=transforms.ToTensor())

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader


class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


def train_model(model, train, test, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        model.train()

        for i, (data, target) in enumerate(train):
            optimizer.zero_grad()
            output = model(data.flatten(start_dim=1))
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}')

        print(f'Epoch {epoch+1} completed')
        print('Testing model...')
        test_model(model, test, loss_fn)
        print('')


def plot_uncertainty(certainty_bins):
    accuracies = [bin['correct'] / bin['total']
                  if bin['total'] > 0 else 0 for bin in certainty_bins]
    bins = [f'{i*10}-{(i+1)*10}%' for i in range(len(certainty_bins))]

    plt.figure(figsize=(10, 6))
    plt.bar(bins, accuracies, color='skyblue')
    plt.xlabel('Certainty Range (%)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Certainty Range')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y')

    plt.show()


@torch.no_grad()
def test_model(model, test, loss_fn):
    model.eval()

    test_loss = 0
    correct = 0

    certainty_bins = [{'correct': 0, 'total': 0} for _ in range(10)]

    for data, target in test:
        input = model(data.flatten(start_dim=1))
        prob = nn.functional.softmax(input, dim=1)

        pred = input.argmax(dim=1, keepdim=True)
        correct_preds = pred.eq(target.view_as(pred))

        for i, true_label in enumerate(target):
            p = prob[i, true_label].item()
            idx = min(int(p * 10), 9)

            certainty_bins[idx]['total'] += 1

            if correct_preds[i]:
                certainty_bins[idx]['correct'] += 1

        test_loss += loss_fn(input, target).item()
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test.dataset)
    acc = 100. * correct / len(test.dataset)
    print(
        f'Average loss: {test_loss: .4f}, Accuracy: {correct}/{len(test.dataset)} ({acc: .2f} %)')

    print(f'Certainty bins: {certainty_bins}')
    plot_uncertainty(certainty_bins)


@torch.no_grad()
def add_noise(model, sigma):
    noisy_model = LogisticRegression(28 * 28, 10)
    noisy_model.load_state_dict(model.state_dict())

    for param in noisy_model.parameters():
        param += torch.randn_like(param) * sigma

    return noisy_model


def main():
    n_inputs = 28 * 28
    n_outputs = 10

    train, test = load_data()

    model = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train, test, optimizer, criterion, epochs=1)

    noise_model = add_noise(model, 500)
    test_model(noise_model, test, criterion)


if __name__ == '__main__':
    main()
