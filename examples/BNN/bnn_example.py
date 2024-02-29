from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions import Normal, Categorical
import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output


def model(x_data, y_data):
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight),
                        scale=torch.ones_like(net.fc1.weight)).to_event(2)
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias),
                        scale=torch.ones_like(net.fc1.bias)).to_event(1)

    outw_prior = Normal(loc=torch.zeros_like(net.out.weight),
                        scale=torch.ones_like(net.out.weight)).to_event(2)
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias),
                        scale=torch.ones_like(net.out.bias)).to_event(1)

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'out.weight': outw_prior, 'out.bias': outb_prior}
    lifted_module = pyro.random_module("module", net, priors)
    lifted_reg_model = lifted_module()
    lhat = log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat).to_event(1), obs=y_data)


def guide(x_data, y_data):
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior.to_event(2), 'fc1.bias': fc1b_prior.to_event(
        1), 'out.weight': outw_prior.to_event(2), 'out.bias': outb_prior.to_event(1)}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../extra/datasets', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),])),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../extra/datasets', train=False, transform=transforms.Compose([transforms.ToTensor(),])
                   ),
    batch_size=128, shuffle=True)

net = NN(28*28, 1024, 10)
optim = Adam({"lr": 0.01})
log_softmax = nn.LogSoftmax(dim=1)
softplus = torch.nn.Softplus()
svi = SVI(model, guide, optim, loss=Trace_ELBO())
num_iterations = 5
loss = 0

for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        loss += svi.step(data[0].view(-1, 28*28), data[1])
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    print("Epoch ", j, " Loss ", total_epoch_loss_train)

num_samples = 10


def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)


print('Prediction when network is forced to predict')
correct = 0
total = 0

for j, data in enumerate(test_loader):
    images, labels = data
    predicted = predict(images.view(-1, 28*28))
    total += labels.size(0)
    predicted = torch.tensor(predicted)
    correct += (predicted == labels).sum().item()
print("accuracy: %d %%" % (100 * correct / total))
