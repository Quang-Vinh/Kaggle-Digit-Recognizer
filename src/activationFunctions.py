import numpy as np
from scipy.stats import logistic

def activation(z):
    return leakyRelu(z)

def activationGradient(z):
    return leakyReluGradient(z)

def outputActivation(z):
    return softMax(z)

def softMax(z):
    t = np.exp(z)
    return t / np.sum(t, axis = 0, keepdims = True)

def tanh(z):
    return np.tanh(z)

def tanhGradient(z):
    return 1 - np.tanh(z) ** 2

def sigmoid(z):
    return logistic.cdf(z)

def sigmoidGradient(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return (z > 0) * z

def reluGradient(z):
    return (z > 0) * 1

def leakyRelu(z):
    return (z > 0) * z + (z <= 0) * z * 0.01

def leakyReluGradient(z):
    return (z > 0) * 1 + (z <= 0) * 0.01 