"""
Main module of smolnet.
"""
from __future__ import annotations
import random

import numpy as np


class Network:
    def __init__(self, size: tuple[int, int, int], cost_fun: str = None):
        self.size = size
        self.num_layer = len(size)
        self.cost_fun = cost_fun
        self.regularization = False

        self.weights = [np.random.randn(size[i], size[i-1]) for i in range(1, self.num_layer)]
        self.biases = [np.random.randn(size[i], 1) for i in range(1, self.num_layer)]

        self.gradient_w = [np.zeros(weight.shape) for weight in self.weights]
        self.gradient_b = [np.zeros(bias.shape) for bias in self.biases]

        self.z = []  #  = Wx + b
        self.a = []  # sigmoid(z)
        self.delta = [np.zeros(layer_size) for layer_size in size[1:]]  # = dC / dz

        self.accuracies = []
        self.learning_rate = 1
    
    def regularize(self, train_size: int, _type: str, _lambda: float = 0.1):
        """Apply regularization to the network.

        Args:
            train_size (int): Size of the training set.
            type (str): Type of regularization; either `l1` or `l2`.
            _lambda (float): The regularization parameter. Defaults to 0.1.
        """
        self.regularization = True
        self.train_size= train_size
        self.reg_type = _type
        self._lambda = _lambda

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
        if self.cost_fun == 'cross-entropy':
            self.delta[-1] = self.a[-1] - y
        else:
            self.delta[-1] = self._cost_derivative(self.a[-1], y) * sigmoid_prime(self.z[-1])

        for l in range(2, self.num_layer):
            self.delta[-l] = (
                (self.weights[-l+1].T @ self.delta[-l+1]) * sigmoid_prime(self.z[-l])
            )

    def backprop(self):
        """Update the gradient of weights and biases.
        """
        for l in range(1, self.num_layer):
            self.gradient_b[-l] =  self.delta[-l].sum(axis=1).reshape(self.biases[-l].shape)
            self.gradient_w[-l] = self.delta[-l] @ self.a[-l-1].T

    def update_params(self, mini_batch_length: int):
        """Update weights and biases after each pass of a mini-batch.

        Args:
            mini_batch_length (int): Number of training samples in the mini-batch.
        """
        eta = self.learning_rate
        for layer in range(self.num_layer - 1):
            self.biases[layer] -= (eta / mini_batch_length) * self.gradient_b[layer]

            if self.regularization:
                if self.reg_type == 'l1':
                    self.weights[layer] -= (
                        (eta * self._lambda / self.train_size) * (self.weights[layer] >= 0)
                        - (eta / mini_batch_length) * self.gradient_w[layer]
                    )
                else:
                    self.weights[layer] -= (
                        (eta * self._lambda / self.train_size) * self.weights[layer]
                        - (eta / mini_batch_length) * self.gradient_w[layer]
                    )
            else:
                self.weights[layer] -= (eta / mini_batch_length) * self.gradient_w[layer]

    def zero_grad(self):
        """Set gradients to zero after each mini-batch pass.
        """
        self.gradient_w = [np.zeros(weight.shape) for weight in self.weights]
        self.gradient_b = [np.zeros(bias.shape) for bias in self.biases]

    def _cost_derivative(self, output: np.ndarray, y: np.ndarray):
        """Default cost function used here is mean squared error. Derivative is w.r.t. `a`.
        """
        return output - y

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
                x = np.array([batch[i][0] for i in range(len(batch))]).reshape(len(batch), -1).T
                y = np.array([batch[i][1] for i in range(len(batch))]).reshape(len(batch), -1).T

                self.feed_forward(x, y)
                self.backprop()
                self.update_params(len(batch))
                self.zero_grad()

            if test_data:
                accuracy = self.evaluate(test_data)
                self.accuracies.append(accuracy)
                if epoch % 3 == 0:
                    print(f"Accuracy at epoch {epoch}: {accuracy}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Return sigmoid(x).
    """
    return 1 / (1 + np.e ** (-x))

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """Return derivative of sigmoid(x).
    """
    return sigmoid(x) * (1 - sigmoid(x))
