import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

data = np.loadtxt('data/data2.csv', delimiter=',', skiprows=1)

print(np.isnan(data).any())
print(len(data))

x_train = torch.tensor(data[:, 0:2], dtype=torch.float32)
y_train = torch.tensor(data[:, 2], dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0],[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return F.mse_loss(self.f(x), y)  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 1e-8)
for epoch in range(40000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    # print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01


    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure()  # Create a new figure
ax = fig.add_subplot(projection='3d')  # Add a 3D subplot

# Your existing scatter plot
xs = x_train[:, 0].detach().numpy()  # Extract the first feature (Day Length) and convert to NumPy array
ys = x_train[:, 1].detach().numpy()  # Extract the second feature (Weight) and convert to NumPy array
zs = y_train.detach().numpy()  # Convert the target variable to NumPy array

ax.scatter(xs, ys, zs, marker='o', alpha=0.1)  # Create a scatter plot in 3D space

# Generate the predicted zs from the model
predicted_zs = model.f(x_train).detach().numpy()

# Create a surface plot using predicted zs
ax.plot_trisurf(xs, ys, predicted_zs.flatten(), alpha=0.7, antialiased=True, cmap='plasma')  # The surface plot based on model predictions

# Labels
ax.set_xlabel('Day Length')  # Set the x-axis label
ax.set_ylabel('Weight')  # Set the y-axis label
ax.set_zlabel('Number of Days')  # Set the z-axis label

ax.set_xlim([min(xs)+3, max(xs)-3])
ax.set_ylim([min(ys)+3, max(ys)-3])
ax.set_zlim([min(zs)+3, max(zs)-3])


# Tilt the plot for better view
ax.view_init(elev=10, azim=70)  # Change these numbers to tilt the plot

plt.show()  # Display the figure


