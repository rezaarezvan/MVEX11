import torch
import torch.nn as nn
from torch.optim import SGD


class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2, 8, bias=False)
        self.Matrix2 = nn.Linear(8, 1, bias=False)

    def forward(self, k):
        k = self.Matrix1(k)  # first layer
        k = self.Matrix2(k)  # second
        return k.squeeze()


if __name__ == '__main__':
    x = torch.tensor([[6, 2], [5, 2], [1, 3], [7, 6]]).float()
    y = torch.tensor([1, 5, 2, 5]).float()
    # m1 = nn.Linear(2, 8, bias=False)
    # m2 = nn.Linear(8, 1, bias=False)

    f = MyNN()
    yhat = f(x)
    # print(yhat)
    # print(y)

    loss = nn.MSELoss()
    loss(y, yhat)
    print(torch.mean((y - yhat) ** 2))
    opt = SGD(f.parameters(), lr=0.01)

    # Adjust parameters
    losses = []
    for _ in range(50):
        opt.zero_grad()  # reset
        loss_value = loss(f(x), y)
        loss_value.backward()  # computes gradient
        opt.step()
        losses.append(loss_value.item())

    print(f(x))
    print(losses)
    print(y)
