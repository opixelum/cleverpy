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
        return Logistic.activate(x) * (1 - Logistic.activate(x))

Sigmoid = Logistic  # Alias
