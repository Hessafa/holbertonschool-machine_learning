#!/usr/bin/env python3
"""
Neural network with training visualization
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Single neuron for binary classification
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
        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        A = self.forward_prop(X)
        P = np.where(A >= 0.5, 1, 0)
        return P, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train neuron with logging and optional graph
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        use_logging = verbose or graph

        if use_logging:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)

            if i == 0 or i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)

                if verbose and (i % step == 0 or i == iterations):
                    print(f"Cost after {i} iterations: {cost}")

            if i != iterations:
                self.gradient_descent(X, Y, A, alpha)

        self.__A = A

        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
