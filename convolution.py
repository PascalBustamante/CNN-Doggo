import numpy as np

from layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, filter_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(filter_size, filter_size, input_channels, output_channels)
        self.biases = np.zeros(output_channels)

    def forward(self, inputs):
        batch_size, height, width, channels = inputs.shape
        out_height = (height - self.filter_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.filter_size + 2 * self.padding) // self.stride + 1

        self.inputs = inputs
        self.output = np.zeros((batch_size, out_height, out_width, self.output_channels))

        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(self.output_channels):
                        self.output[i, h, w, c] = np.sum(
                            inputs[i, h*self.stride:h*self.stride+self.filter_size, w*self.stride:w*self.stride+self.filter_size, :]
                            * self.weights[:, :, :, c]
                        ) + self.biases[c]

        return self.output

    def backward(self, gradients):
        batch_size, height, width, channels = self.inputs.shape
        _, out_height, out_width, out_channels = gradients.shape

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        self.dinputs = np.zeros_like(self.inputs)

        for i in range(batch_size):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(out_channels):
                        self.dweights[:, :, :, c] += self.inputs[i, h*self.stride:h*self.stride+self.filter_size, w*self.stride:w*self.stride+self.filter_size, :] * gradients[i, h, w, c]
                        self.dbiases[c] += gradients[i, h, w, c]
                        self.dinputs[i, h*self.stride:h*self.stride+self.filter_size, w*self.stride:w*self.stride+self.filter_size, :] += self.weights[:, :, :, c] * gradients[i, h, w, c]

        return self.dinputs
