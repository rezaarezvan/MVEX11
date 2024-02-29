import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


def load_data():
    trans = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.GTSRB(
        root='../extra/datasets', split='train', download=True, transform=trans)
    testset = torchvision.datasets.GTSRB(
        root='../extra/datasets', split='test', download=True, transform=trans)

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


def train_model(model, train, test, optimizer, epochs=3):
    loss = nn.CrossEntropyLoss()
    acc = 0

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train):
            optimizer.zero_grad()
            input = model(data.view(-1, 28*28*3))
            output = loss(input, target)
            output.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {output.item()}')

        print(f'Epoch {epoch+1} completed')
        print('Testing model...')
        acc = test_model(model, test)
        print('')

    return acc


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


def test_model(model, test):
    loss = nn.CrossEntropyLoss()

    model.eval()

    test_loss = 0
    correct = 0

    certainty_bins = [{'correct': 0, 'total': 0} for _ in range(10)]

    with torch.no_grad():
        for data, target in test:
            input = model(data.view(-1, 28*28*3))
            prob = nn.functional.softmax(input, dim=1)

            pred = input.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(target.view_as(pred))

            for i, true_label in enumerate(target):
                p = prob[i, true_label].item()
                idx = min(int(p * 10), 9)

                certainty_bins[idx]['total'] += 1

                if correct_preds[i]:
                    certainty_bins[idx]['correct'] += 1

            test_loss += loss(input, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test.dataset)
    acc = 100. * correct / len(test.dataset)
    print(
        f'Average loss: {test_loss: .4f}, Accuracy: {correct}/{len(test.dataset)} ({acc: .2f} %)')

    print(f'Certainty bins: {certainty_bins}')

    plot_uncertainty(certainty_bins)

    return acc


def main():
    n_inputs = 3 * 28 * 28
    n_outputs = 43
    train, test = load_data()
    model = LogisticRegression(n_inputs, n_outputs)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    no_noise_acc = train_model(model, train, test, optimizer, epochs=1)

    sigma = 0.5
    noise_weights = torch.rand(43, 28*28*3) * sigma
    for param in model.parameters():
        param.data += noise_weights

    noise_acc = test_model(model, test)

    print(f'Non-noise accuracy: {no_noise_acc}')
    print(f'Noise accuracy: {noise_acc}')
    print(f'Accuracy difference: {no_noise_acc - noise_acc}')


if __name__ == '__main__':
    main()
