#!/usr/bin/env python3
"""
Neural network with one hidden layer for binary classification
"""

import numpy as np


class NeuralNetwork:
    """
    One hidden layer neural network
    """

    def __init__(self, nx, nodes):
        """
        Initialize neural network

        Args:
            nx (int): number of input features
            nodes (int): number of hidden nodes
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Output layer
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
