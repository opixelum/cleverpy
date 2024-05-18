from typing import List
from activation import Activation
import random
from optimizer import Optimizer, BGD
from loss import Loss

class Neuron:
    def __init__(self, num_inputs: int, activation: Activation):
        self.weights = [random.random() * 2 - 1 for _ in range(num_inputs + 1)]  # +1 for bias
        self.output = 0.0
        self.gradient = 0.0
        self.activation = activation

    def activate(self, inputs: List[float]) -> float:
        self.output = self.weights[0]  # Bias
        for i in range(len(inputs)):
            self.output += self.weights[i + 1] * inputs[i]
        self.output = self.activation.activate(self.output)
        return self.output

    def calculate_gradient(self, target: float = None, downstream_gradients: List[float] = None, downstream_weights: List[float] = None, loss: Loss = None) -> float:
        if target is not None:
            self.gradient = self.activation.derivative(self.output) * loss.derivative(self.output, target)
        else:
            self.gradient = sum(downstream_gradient * weight for downstream_gradient, weight in zip(downstream_gradients, downstream_weights))
            self.gradient *= self.activation.derivative(self.output)
        return self.gradient

    def update_weights(self, inputs: List[float], learning_rate: float, optimizer: Optimizer, layer_index: int, neuron_index: int):
        for i in range(len(self.weights)):
            if isinstance(optimizer, BGD):
                optimizer.accumulate_gradient(layer_index, neuron_index, i, self.gradient if i == 0 else self.gradient * inputs[i - 1])
            self.weights[i] = optimizer.update(self.weights[i], self.gradient if i == 0 else self.gradient * inputs[i - 1], learning_rate, layer_index, neuron_index, i)
