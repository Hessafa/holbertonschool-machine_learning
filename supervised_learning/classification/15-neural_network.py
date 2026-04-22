class Neuron:
    """Neuron class that performs binary classification."""

    def forward_prop(self, X):
        """Forward propagation.

        Args:
            X (numpy.ndarray): input data (nx, m)

        Returns:
            numpy.ndarray: activated output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
