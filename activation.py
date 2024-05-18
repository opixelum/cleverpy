import math

class Activation:
    @staticmethod
    def activate(x: float) -> float:
        raise NotImplementedError

    @staticmethod
    def derivative(x: float) -> float:
        raise NotImplementedError

class Tanh(Activation):
    @staticmethod
    def activate(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def derivative(x: float) -> float:
        return 1 - math.tanh(x) ** 2

class Logistic(Activation):  # Alias for sigmoid
    @staticmethod
    def activate(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def derivative(x: float) -> float:
        sigmoid = Logistic.activate(x)
        return sigmoid * (1 - sigmoid)

Sigmoid = Logistic  # Alias