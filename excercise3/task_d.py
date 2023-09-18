import torch
import torch.nn as nn
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
fashion_mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = fashion_mnist_train.data.reshape(-1, 1, 28,
                                           28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((fashion_mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(fashion_mnist_train.targets.shape[0]), fashion_mnist_train.targets] = 1  # Populate output

fashion_mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = fashion_mnist_test.data.reshape(-1, 1, 28,
                                         28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((fashion_mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(fashion_mnist_test.targets.shape[0]), fashion_mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class CnnModel(nn.Module):

    def __init__(self):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(32)  # BatchNorm after Conv
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(64)  # BatchNorm after Conv
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout1 = nn.Dropout(0.5)  # Dropout after first Dense layer
        self.fc2 = nn.Linear(128, 10)

    def logits(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)  # BatchNorm before ReLU
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)  # BatchNorm before ReLU
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(-1, 64 * 7 * 7)

        x = self.fc1(x)
        x = self.dropout1(x)  # Dropout after first Dense layer
        x = self.fc2(x)

        return x

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


def run_model_training():
    model = CnnModel()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    for epoch in range(20):
        for batch in range(len(x_train_batches)):
            model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
            optimizer.step()  # Perform optimization by adjusting W and b,
            optimizer.zero_grad()  # Clear gradients for next step
        print(epoch, "accuracy = %s" % model.accuracy(x_test, y_test))


run_model_training()
