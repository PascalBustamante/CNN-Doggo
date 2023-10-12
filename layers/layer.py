import numpy as np


class Layer:
    def __init__(self) -> None:
        pass

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learn_rate):
        raise NotImplementedError
