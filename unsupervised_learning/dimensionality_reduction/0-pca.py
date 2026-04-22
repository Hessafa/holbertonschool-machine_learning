#!/usr/bin/env python3
"""
Function that performs PCA on a dataset.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X.

    Args:
        X (numpy.ndarray): shape (n, d)
        var (float): fraction of variance to retain

    Returns:
        W (numpy.ndarray): projection matrix
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    variance = S ** 2
    explained_variance_ratio = variance / np.sum(variance)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # ✅ FIX HERE
    nd = np.argmax(cumulative_variance >= var) + 1

    W = Vt[:nd].T

    return W
