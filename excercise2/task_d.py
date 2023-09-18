import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output


class numberModel:

    def __init__(self):
        self.W = torch.rand(784, 10, requires_grad=True)
        self.b = torch.rand(1, 10, requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return F.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = numberModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.01)

for epoch in range(1000):
    model.loss(x_train, y_train).backward()
    if epoch % 100 == 0:
        print("epoch = %s, loss = %s" % (epoch, model.loss(x_train, y_train)))
        print(epoch, model.accuracy(x_test, y_test))
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

print("accuracy = %s" % model.accuracy(x_test, y_test))

# Assuming model.W is your trained weight matrix of shape [784, 10]
W_numpy = model.W.detach().numpy()  # Convert the tensor to numpy array and detach it from the computation graph

for i in range(10):  # Loop through each of the 10 digits
    digit_weights = W_numpy[:, i]  # Extract the weights corresponding to the i-th digit
    digit_image = digit_weights.reshape(28, 28)  # Reshape it back to 28x28

    plt.imshow(digit_image, cmap='gray')  # Display the image
    plt.axis('off')  # Turn off axis
    plt.savefig(f'digit_{i}.png')  # Save the image
    plt.close()  # Close the plot to free up resources
