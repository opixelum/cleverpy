class Optimizer:
    def update(self, weight: float, gradient: float, learning_rate: float, layer_index: int = None, neuron_index: int = None, weight_index: int = None) -> float:
        raise NotImplementedError

class SGD(Optimizer):
    def update(self, weight: float, gradient: float, learning_rate: float, layer_index: int = None, neuron_index: int = None, weight_index: int = None) -> float:
        return weight - learning_rate * gradient

class BGD(Optimizer):
    def __init__(self):
        self.accumulated_gradients = {}

    def accumulate_gradient(self, layer_index: int, neuron_index: int, weight_index: int, gradient: float):
        key = (layer_index, neuron_index, weight_index)
        if key not in self.accumulated_gradients:
            self.accumulated_gradients[key] = 0.0
        self.accumulated_gradients[key] += gradient

    def update(self, weight: float, gradient: float, learning_rate: float, layer_index: int = None, neuron_index: int = None, weight_index: int = None) -> float:
        key = (layer_index, neuron_index, weight_index)
        if key in self.accumulated_gradients:
            avg_gradient = self.accumulated_gradients[key] / len(self.accumulated_gradients)
            weight -= learning_rate * avg_gradient
            del self.accumulated_gradients[key]
        return weight
