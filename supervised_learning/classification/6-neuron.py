#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""

import numpy as np


class Neuron:
    """
    Neuron class for binary classification
    """

    def __init__(self, nx):
        """
        Initialize neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights"""
        return self.__W

    @property
    def b(self):
        """Bias"""
        return self.__b

    @property
    def A(self):
        """Activation"""
        return self.__A

    def forward_prop(self, X):
        """
        Forward propagation
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Compute cost
        """
        m = Y.shape[1]

        return -(1 / m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """
        Evaluate neuron
        """
        A = self.forward_prop(X)
        P = np.where(A >= 0.5, 1, 0)
        return P, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        One step gradient descent
        """
        m = Y.shape[1]

        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train neuron

        Args:
            X (ndarray): input data
            Y (ndarray): labels
            iterations (int): training iterations
            alpha (float): learning rate

        Returns:
            tuple: (predictions, cost)

        Raises:
            TypeError / ValueError as specified
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
