class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradients):
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
        return gradients

    def train(self, inputs, targets, learning_rate):
        self.forward(inputs)
        gradients = self.backward(targets)
        for layer in self.layers:
            layer.update(learning_rate)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
