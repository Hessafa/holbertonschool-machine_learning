#!/usr/bin/env python3
"""
Function that performs PCA on a dataset.
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on dataset X and reduces its dimensionality.

    Args:
        X (numpy.ndarray): shape (n, d)
        ndim (int): new dimensionality

    Returns:
        T (numpy.ndarray): transformed dataset (n, ndim)
    """
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Step 3: Select top ndim components
    W = Vt[:ndim].T  # shape (d, ndim)

    # Step 4: Project data
    T = np.matmul(X_centered, W)

    return T
