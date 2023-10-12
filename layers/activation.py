import numpy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activation_function) -> None:
        super().__init__()
        self.actication_function = activation_function

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.actication_function(inputs)
        return self.output

    def backward(self, gradients):
        self.dinputs = gradients * self.actication_function.derivative(self.inputs)
        return self.dinputs
