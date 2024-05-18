from model import Model
from activation import Tanh, Sigmoid
from loss import MSE, LogLoss
from optimizer import SGD, BGD

# XOR data
X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

Y = [
    [-1.0],  # Expected output for XOR with logistic
    [1.0],
    [1.0],
    [-1.0]
]

# Creating Model with Sigmoid activation function
activation = Tanh()
model = Model([2, 3, 1], activation)

# Define the loss function and optimizer
loss = MSE()
optimizer = SGD()

# Training the model with logistic activation and log loss
model.train(X, Y, 0.1, 10000, loss, optimizer)

# Making predictions
predictions = [model.predict(X[k]) for k in range(len(X))]
for i, prediction in enumerate(predictions):
    print(f"Input: {X[i]}, Prediction: {prediction}, Expected: {Y[i]}")
