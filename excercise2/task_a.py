import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


x_train = torch.tensor([[0.0], [1.0]], dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor([[1], [0]], dtype=torch.float)

class notModel:

    def __init__(self):
        self.W = torch.rand(1, 1, requires_grad=True)
        self.b = torch.rand(1, 1, requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return F.binary_cross_entropy_with_logits(self.logits(x), y)

model = notModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.1)

for epoch in range(100000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')

# Add this line to plot the sigmoid curve for a range of x values
x_range = torch.arange(0.0, 1.0, 0.001).reshape(-1, 1)
plt.plot(x_range, model.f(x_range).detach().numpy(), label='Sigmoid Curve')

plt.legend()
plt.show()
