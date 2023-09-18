import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

data = np.loadtxt('data/data3.csv', delimiter=',', skiprows=1)

print(np.isnan(data).any())
print(len(data))

x_train = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data[:, 1], dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.randn([1, 1], requires_grad=True)
        self.b = torch.randn([1], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20*torch.sigmoid(x @ self.W + self.b) +31  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return F.mse_loss(self.f(x), y)  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W, model.b], 1e-3)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    if epoch % 1000 == 0 or epoch in range(10):
        print(f'Epoch {epoch}, Loss: {model.loss(x_train, y_train)}')
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01


    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
x_vals = torch.linspace(torch.min(x_train), torch.max(x_train), 100).reshape(-1, 1)
y_vals = model.f(x_vals).detach()

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$', markersize=4, alpha=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_vals, y_vals, label='$f(x) = 20\sigma(xW+b) + 31$', linewidth=2, color='red')
plt.legend()
plt.show()

