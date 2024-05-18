from typing import List
from layer import Layer, Dense
from loss import Loss
from optimizer import Optimizer, BGD

class Model:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self._initialize_layers()

    def _initialize_layers(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                # Assume the input layer size is equal to the number of inputs
                num_inputs = layer.num_neurons
            else:
                num_inputs = self.layers[i-1].num_neurons
            if isinstance(layer, Dense):
                layer.initialize(num_inputs)

    def _forward_propagate(self, inputs: List[float]) -> List[float]:
        for layer in self.layers:
            inputs = layer.activate(inputs)
        return inputs

    def _backward_propagate(self, targets: List[float], loss: Loss):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:  # Output layer
                layer.calculate_gradients(targets=targets, loss=loss)
            else:  # Hidden layer
                downstream_layer = self.layers[i + 1]
                layer.calculate_gradients(downstream_layer=downstream_layer)

    def _update_weights(self, inputs: List[float], learning_rate: float, optimizer: Optimizer):
        for layer_index, layer in enumerate(self.layers):
            layer.update_weights(inputs, learning_rate, optimizer, layer_index)
            inputs = [neuron.output for neuron in layer.neurons]

    def fit(self, all_samples_inputs: List[List[float]], all_samples_expected_outputs: List[List[float]], learning_rate: float, epochs: int, loss: Loss, optimizer: Optimizer):
        for epoch in range(epochs):
            for k in range(len(all_samples_inputs)):
                sample_inputs = all_samples_inputs[k]
                sample_expected_outputs = all_samples_expected_outputs[k]

                # Forward pass
                self._forward_propagate(sample_inputs)
                # Backward pass
                self._backward_propagate(sample_expected_outputs, loss)
                # Update weights
                self._update_weights(sample_inputs, learning_rate, optimizer)

            # Apply batch updates if using BGD
            if isinstance(optimizer, BGD):
                for k in range(len(all_samples_inputs)):
                    sample_inputs = all_samples_inputs[k]
                    self._update_weights(sample_inputs, learning_rate, optimizer)

    def predict(self, inputs: List[float]) -> List[float]:
        return self._forward_propagate(inputs)
