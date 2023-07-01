"""
Main module of smolnet.
"""
from __future__ import annotations
import random

import numpy as np


class Network:
    def __init__(self, size: tuple):
        self.size = size
        self.num_layer = len(size)

        self.weights = [np.random.randn(size[i], size[i-1]) for i in range(1, self.num_layer)]
        self.biases = [np.random.randn(size[i], 1) for i in range(1, self.num_layer)]

        self.gradient_w = [np.zeros(weight.shape) for weight in self.weights]
        self.gradient_b = [np.zeros(bias.shape) for bias in self.biases]

        self.z = []  #  = Wx + b
        self.a = []  # sigmoid(z)
        self.delta = [np.zeros(layer_size) for layer_size in size[1:]]  # = dC / dz

        self.accuracies = []
        self.learning_rate = 1

    def feed_forward(self, a_l: np.ndarray, y: np.ndarray):
        """Feedforward each input and calculate deltas at each layer which are to be used later in
        backpropagation.

        Args:
            a_l (np.ndarray): The input vector.
            y (np.ndarray): The output vector (prediction).
        """
        self.z = []
        self.a = []

        self.a.append(a_l)  # <- input layer

        for l in range(self.num_layer - 1):  # last layer is output layer
            z_l = np.dot(self.weights[l], a_l)
            a_l = sigmoid(z_l + self.biases[l])

            self.a.append(a_l)
            self.z.append(z_l)

        # delta_L, of last layer
        self.delta[-1] = self._cost_derivative(self.a[-1], y) * sigmoid_prime(self.z[-1])

        for l in range(2, self.num_layer):
            self.delta[-l] = (
                np.dot(self.weights[-l+1].T, self.delta[-l+1]) * sigmoid_prime(self.z[-l])
            )

    def backprop(self):
        """Update the gradient of weights and biases.
        """
        for l in range(1, self.num_layer):
            self.gradient_b[-l] +=  self.delta[-l]
            self.gradient_w[-l] += self.delta[-l].reshape(-1, 1) @ self.a[-l-1].reshape(-1, 1).T

    def update_params(self, mini_batch_length: int):
        """Update weights and biases after each pass of a mini-batch.

        Args:
            mini_batch_length (int): Number of training samples in the mini-batch.
        """
        eta = self.learning_rate
        for layer in range(self.num_layer - 1):
            self.weights[layer] -= (eta / mini_batch_length) * self.gradient_w[layer]
            self.biases[layer] -= (eta / mini_batch_length) * self.gradient_b[layer]

    def zero_grad(self):
        """Set gradients to zero after each mini-batch pass.
        """
        self.gradient_w = [np.zeros(weight.shape) for weight in self.weights]
        self.gradient_b = [np.zeros(bias.shape) for bias in self.biases]

    def _cost_derivative(self, output: np.ndarray, y: np.ndarray):
        """Cost function used here is mean squared error."""
        return output - y

    def train(self, training_data: list[tuple], epochs: int, batch_size: int, test_data = None):
        """Train the network and calculate accuracy after each epoch.

        Args:
            training_data (list[tuple]): List containing all training sample, in this format - 
                 [(x1, y1), (x2, y2), ..., (x_n, y_n)]
            epochs (int): Number of epochs to train the network for.
            batch_size (int): Number of training samples in each minibatch.
            test_data ((list[tuple]), optional): Test data to calculate accuracy post training.
                Defaults to None.
        """
        for epoch in range(1, epochs + 1):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)
            ]

            for batch in mini_batches:
                x = [batch[i][0] for i in range(len(batch))]
                y = [batch[i][1] for i in range(len(batch))]

                for x_i, y_i in zip(x, y):
                    self.feed_forward(x_i, y_i)
                    self.backprop()

                self.update_params(len(batch))
                self.zero_grad()

            if test_data:
                accuracy = self.evaluate(test_data)
                self.accuracies.append(accuracy)
                if epoch % 3 == 0:
                    print(f"Accuracy at epoch {epoch}: {accuracy}")

    def predict(self, x: np.ndarray):
        """Return predicted value based on input.

        Args:
            x (np.ndarray): Input to the network.

        Returns:
            np.ndarray: Output of the model.
        """
        for l in range(self.num_layer - 1):
            z_l = np.dot(self.weights[l], x)
            x = sigmoid(z_l + self.biases[l])
        return x

    def evaluate(self, test_data: list[tuple]) -> float:
        """Evaluate the network with test data.

        Args:
            test_data (list[tuple]): List of test samples of this format -
                [(x1, y1), (x2, y2), ..., (x_n, y_n)]

        Returns:
            float: Accuracy on test data.
        """
        predicted_labels = [np.argmax(self.predict(x)) for x, _ in test_data]
        actual_lebels = [np.argmax(y) for _, y in test_data]
        num_correct_labels =  sum(
            [predicted_labels[i] == actual_lebels[i] for i in range(len(test_data))]
        )
        return num_correct_labels / len(test_data)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Return sigmoid(x).
    """
    return 1 / (1 + np.e ** (-x))

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """Return derivative of sigmoid(x).
    """
    return sigmoid(x) * (1 - sigmoid(x))
