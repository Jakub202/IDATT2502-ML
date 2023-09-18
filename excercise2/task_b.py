import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float)

class nandModel:

    def __init__(self):
        self.W = torch.rand(2, 1, requires_grad=True)
        self.b = torch.rand(1, 1, requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return F.binary_cross_entropy_with_logits(self.logits(x), y)

model = nandModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.1)

for epoch in range(100000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the training data
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, label='Training Data', c='blue')

# Create a meshgrid to evaluate the model
x1 = np.linspace(0, 1.5, 30)  # adjust the range and number of points as needed
x2 = np.linspace(0, 1.5, 30)
x1, x2 = np.meshgrid(x1, x2)
x_mesh = torch.tensor(np.column_stack((x1.ravel(), x2.ravel())), dtype=torch.float)

# Evaluate the model on the meshgrid
y_mesh = model.f(x_mesh).detach().numpy().reshape(x1.shape)

# Plot the model's surface
ax.plot_surface(x1, x2, y_mesh, alpha=0.3, label='Model Surface')

# Labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('NAND Gate Model')

plt.show()


