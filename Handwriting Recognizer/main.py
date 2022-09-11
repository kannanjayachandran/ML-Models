# Two layer neuron for digit recognition using MNIST dataset.

import numpy as np
import pandas as pd


# Read the data
data = pd.read_csv('data/train.csv')
data.head()

# Data wrangling
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255.

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.
_, m_train = x_train.shape

print(y_train)


# Code for the actual neural network

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def re_lu(z):
    return np.maximum(z, 0)


def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a


def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = re_lu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def re_lu_derv(z):
    return z > 0


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def backward_prop(z1, a1, z2, a2, w1, w2, x, y):
    one_hot_y = one_hot(y)
    d_z2 = a2 - one_hot_y
    d_w2 = 1 / m * d_z2.dot(a1.T)
    db2 = 1 / m * np.sum(d_z2)
    d_z1 = w2.T.dot(d_z2) * re_lu_derv(z1)
    d_w1 = 1 / m * d_z1.dot(x.T)
    db1 = 1 / m * np.sum(d_z1)
    return d_w1, db1, d_w2, db2


def update_params(w1, b1, w2, b2, d_w1, db1, d_w2, db2, alpha):
    w1 = w1 - alpha * d_w1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * d_w2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def gradient_descent(x, y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        d_w1, db1, d_w2, db2 = backward_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, d_w1, db1, d_w2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2


W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, 500)
