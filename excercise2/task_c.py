import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

x_train = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float).reshape(-1, 2)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)


converge = True

if converge:
    W1_init = torch.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True)
    b1_init = torch.tensor([[-5.0, 15.0]], requires_grad=True)
    W2_init = torch.tensor([[10.0], [10.0]], requires_grad=True)
    b2_init = torch.tensor([[-15.0]], requires_grad=True)
else:
    W1_init = torch.tensor([[7.0, -7.0], [7.0, -7.0]], requires_grad=True)
    b1_init = torch.tensor([[-3.0, 4.0]], requires_grad=True)
    W2_init = torch.tensor([[7.0], [-7.0]], requires_grad=True)
    b2_init = torch.tensor([[3.0]], requires_grad=True)



class xorModel:

    def __init__(self, W1=W1_init, W2=W2_init, b1=b1_init, b2=b2_init):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self, h):
        return h @ self.W2 + self.b2

    def f(self, x):
        return torch.sigmoid(self.f2(self.f1(x)))

    def loss(self, x, y):
        return F.binary_cross_entropy_with_logits(self.f2(self.f1(x)), y)



model = xorModel()

optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], 0.1)

for epoch in range(100000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" % (
model.W1, model.b1, model.W2, model.b2, model.loss(x_train, y_train)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the training data
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, label='Training Data', c='blue')

# Create a meshgrid to evaluate the model
x1 = np.linspace(-0.5, 1.5, 30)  # adjust the range and number of points as needed
x2 = np.linspace(-0.5, 1.5, 30)
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
ax.set_title('XOR Gate Model')

plt.show()
