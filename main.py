from model import Model
from layer import Dense
from activation import *
from loss import *
from optimizer import *

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
model = Model([
    Dense(2, activation),  # input layer with 2 neurons
    Dense(3, activation),  # hidden layer with 3 neurons
    Dense(1, activation)   # output layer with 1 neuron
])

# Define the loss function and optimizer
loss = MSE()
optimizer = SGD()

# Training the model with logistic activation and log loss
model.fit(X, Y, 0.1, 10000, loss, optimizer)

# Making predictions
predictions = [model.predict(X[k]) for k in range(len(X))]
for i, prediction in enumerate(predictions):
    print(f"Input: {X[i]}, Prediction: {prediction}, Expected: {Y[i]}")
