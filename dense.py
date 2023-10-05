from layer import Layer
import numpy as np


class DenseLayer(Layer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, gradients):
        self.dweights = np.dot(self.inputs.T, gradients)
        self.dbiases = np.sum(gradients, axis=0)
        self.dinputs = np.dot(gradients, self.weights.T)
        return self.dinputs