#!/usr/bin/env python3
"""
Function that performs PCA on a dataset.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X.

    Args:
        X (numpy.ndarray): shape (n, d), centered data
        var (float): fraction of variance to retain

    Returns:
        W (numpy.ndarray): shape (d, nd), projection matrix
    """
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance
    variance = (S ** 2)
    total_variance = np.sum(variance)
    explained_variance_ratio = variance / total_variance

    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find number of dimensions to keep
    nd = np.searchsorted(cumulative_variance, var) + 1

    # Projection matrix
    W = Vt[:nd].T

    return W
