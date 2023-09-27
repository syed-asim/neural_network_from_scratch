import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(1000, 3)

np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
layer1.forward(X)
print(layer1.output)

activation1 = Activation_ReLU()
activation1.forward(layer1.output)

print('----')
print(activation1.output)
