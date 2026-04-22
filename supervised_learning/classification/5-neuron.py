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

        Args:
            nx (int): number of input features

        Raises:
            TypeError: nx must be a integer
            ValueError: nx must be positive
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
        """Weights vector"""
        return self.__W

    @property
    def b(self):
        """Bias"""
        return self.__b

    @property
    def A(self):
        """Activated output"""
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
        Computes cost
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
        One step of gradient descent

        Args:
            X (ndarray): input data (nx, m)
            Y (ndarray): correct labels (1, m)
            A (ndarray): activated output (1, m)
            alpha (float): learning rate
        """
        m = Y.shape[1]

        dZ = A - Y

        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
