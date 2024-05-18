import math

class Loss:
    @staticmethod
    def loss(predicted: float, target: float) -> float:
        raise NotImplementedError

    @staticmethod
    def derivative(predicted: float, target: float) -> float:
        raise NotImplementedError

class MSE(Loss):
    @staticmethod
    def loss(predicted: float, target: float) -> float:
        return 0.5 * (predicted - target) ** 2

    @staticmethod
    def derivative(predicted: float, target: float) -> float:
        return predicted - target

class LogLoss(Loss):
    @staticmethod
    def loss(predicted: float, target: float) -> float:
        # Add epsilon to prevent log(0)
        epsilon = 1e-15
        predicted = max(min(predicted, 1 - epsilon), epsilon)
        return -target * math.log(predicted) - (1 - target) * math.log(1 - predicted)

    @staticmethod
    def derivative(predicted: float, target: float) -> float:
        # Add epsilon to prevent division by zero
        epsilon = 1e-15
        predicted = max(min(predicted, 1 - epsilon), epsilon)
        return (predicted - target) / (predicted * (1 - predicted))
