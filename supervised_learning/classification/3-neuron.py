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
        Initialize the neuron

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
        Calculates forward propagation of the neuron

        Args:
            X (numpy.ndarray): shape (nx, m) input data

        Returns:
            numpy.ndarray: activated output (A)
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates logistic regression cost

        Args:
            Y (numpy.ndarray): correct labels (1, m)
            A (numpy.ndarray): activated output (1, m)

        Returns:
            float: cost of the model
        """
        m = Y.shape[1]

        cost = -(1 / m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        )

        return cost
