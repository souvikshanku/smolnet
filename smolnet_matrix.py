"""
Main module of smolnet.
"""
from __future__ import annotations
import random

import numpy as np


class Network:
    def __init__(self, size: tuple[int, int, int], cost_fun: str=None):
        self.size = size
        self.num_layer = len(size)
        self.cost_fun = cost_fun
        self.regularization = False
        self._dropout = False
        self._optim_AdamW = False

        self.weights = [np.random.randn(size[i], size[i-1]) for i in range(1, self.num_layer)]
        self.biases = [np.random.randn(size[i], 1) for i in range(1, self.num_layer)]

        self.gradient_w = [np.zeros(weight.shape) for weight in self.weights]
        self.gradient_b = [np.zeros(bias.shape) for bias in self.biases]

        self.z = []  #  = Wx + b
        self.a = []  # sigmoid(z)
        self.delta = [np.zeros(layer_size) for layer_size in size[1:]]  # = dC / dz

        self.accuracies = []
        self.learning_rate = 1

    def dropout(self, dropout_prob: float=0.8):
        """Apply dropout in the network.

        Args:
            dropout_prob (float): Dropout probability. Defaults to 0.8.
        """
        self._dropout = True
        self.dropout_prob = dropout_prob
    
    def regularize(self, train_size: int, _type: str, _lambda: float=0.1):
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

    def optim_AdamW(
            self, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-08, weight_decay: float=0
        ):
        """Apply AdamW optimization (not the l1/l2 regularized one) in sgd. Supposed to be used
        with baked in L2 regularization, not with the stand-alone one.

        Args:
            beta1 (float, optional): _description_. Defaults to 0.9.
            beta2 (float, optional): _description_. Defaults to 0.999.
            epsilon (float, optional): _description_. Defaults to 1e-08.
            weight_decay (float, optional): _description_. Defaults to 0.
        """
        self._optim_AdamW = True
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.mov_avg_m = [np.zeros(weight.shape) for weight in self.weights]  # = m_t
        self.mov_avg_v = [np.zeros(weight.shape) for weight in self.weights]  # = v_t

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

            if self._dropout and l != self.num_layer - 2:  # don't dropout the output layer!
                r = np.random.binomial(1, 0.5, size=a_l.shape)
                a_l = a_l * r

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

            if self._optim_AdamW:
                self.mov_avg_m[-l] += (
                    self.beta1 * self.mov_avg_m[-l]
                    + (1 - self.beta1) * self.gradient_w[-l]
                )

                self.mov_avg_v[-l] += (
                    self.beta2 * self.mov_avg_v[-l]
                    + (1 - self.beta2) * self.gradient_w[-l] ** 2
                )

    def update_params_w_AdamW(self, mini_batch_length: int):
        """Update weights and biases when AdamW optimization is used.

        Args:
            mini_batch_length (int): Number of training samples in the mini-batch.
        """
        lr = self.learning_rate
        for layer in range(self.num_layer - 1):
            self.biases[layer] -= (lr / mini_batch_length) * self.gradient_b[layer]

            m_hat = self.mov_avg_m[layer] / (1 - self.beta1 ** mini_batch_length)
            v_hat = self.mov_avg_v[layer] / (1 - self.beta2 ** mini_batch_length)
            mov_avg = m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.weights[layer] -= (lr / mini_batch_length) * (mov_avg + self.weight_decay * self.gradient_w[layer])

    def update_params(self, mini_batch_length: int):
        """Update weights and biases after each pass of a mini-batch.

        Args:
            mini_batch_length (int): Number of training samples in the mini-batch.
        """
        lr = self.learning_rate
        for layer in range(self.num_layer - 1):
            self.biases[layer] -= (lr / mini_batch_length) * self.gradient_b[layer]

            if self.regularization:
                if self.reg_type == 'l1':
                    self.weights[layer] = (
                        self.weights[layer]
                        - (lr * self._lambda / self.train_size) * (self.weights[layer] >= 0)
                        - (lr / mini_batch_length) * self.gradient_w[layer]
                    )
                else:
                    self.weights[layer] = (
                        self.weights[layer]
                        - (lr * self._lambda / self.train_size) * self.weights[layer]
                        - (lr / mini_batch_length) * self.gradient_w[layer]
                    )
            else:
                self.weights[layer] -= (lr / mini_batch_length) * self.gradient_w[layer]

    def zero_grad(self):
        """Set gradients to zero after each mini-batch pass.
        """
        self.gradient_w = [np.zeros(weight.shape) for weight in self.weights]
        self.gradient_b = [np.zeros(bias.shape) for bias in self.biases]

        if self.optim_AdamW:
            self.mov_avg_m = [np.zeros(weight.shape) for weight in self.weights]
            self.mov_avg_v = [np.zeros(weight.shape) for weight in self.weights]

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
            if self._dropout:
                z_l = np.dot(self.weights[l] * self.dropout_prob, x)
            else:
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

    def train(self, training_data: list[tuple], epochs: int, batch_size: int, test_data=None):
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
                if self._optim_AdamW:
                    self.update_params_w_AdamW(len(batch))
                else:
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
