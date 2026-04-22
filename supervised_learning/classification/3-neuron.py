#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""

import numpy as np


class Neuron:
    """
    Neuron class
    """

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Computes logistic regression cost

        Args:
            Y (numpy.ndarray): correct labels (1, m)
            A (numpy.ndarray): activated output (1, m)

        Returns:
            cost (float)
        """
        m = Y.shape[1]

        cost = -(1 / m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        )

        return cost
