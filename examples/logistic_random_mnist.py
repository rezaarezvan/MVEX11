import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)

train_dataset = datasets.MNIST(
    root='../extra/datasets', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root='../extra/datasets', train=False, transform=transforms.ToTensor())

print("Number of training samples: " + str(len(train_dataset)) + "\n" +
      "Number of test samples: " + str(len(test_dataset)))

batch_size = 32
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)



class LogisticRegression(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, sigma=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn(28**2) + self.sigma
        x    += x + noise
        return self.linear(x)


n_inputs = 28 * 28
n_outputs = 10
log_regr = LogisticRegression(n_inputs, n_outputs)
optimizer = torch.optim.Adam(log_regr.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10
Loss = []
acc = []
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = log_regr(images.view(-1, 28*28))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        Loss.append(loss.item())

    log_regr.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            outputs = log_regr(images.view(-1, 28*28))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
        accuracy = 100 * (correct.item()) / len(test_dataset)
        acc.append(accuracy)
    print('Epoch: {}. Last Batch Loss: {}. Accuracy: {}'.format(
        epoch, Loss[-1], accuracy))
    log_regr.train()

plt.plot(Loss)
plt.xlabel("Number of Batches")
plt.ylabel("Loss")
plt.title("Training Loss per Batch")
plt.show()

plt.plot(acc)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy per Epoch")
plt.show()
