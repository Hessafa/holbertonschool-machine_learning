#!/usr/bin/env python3
"""
Function that initializes variables for t-SNE.
"""

import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables for calculating P affinities in t-SNE.

    Args:
        X (numpy.ndarray): shape (n, d)
        perplexity (float)

    Returns:
        D (numpy.ndarray): squared distance matrix (n, n)
        P (numpy.ndarray): zeros matrix (n, n)
        betas (numpy.ndarray): ones (n, 1)
        H (float): Shannon entropy
    """
    n = X.shape[0]

    # Compute squared pairwise distances efficiently
    sum_X = np.sum(X ** 2, axis=1)
    D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)

    # Ensure diagonal is zero
    np.fill_diagonal(D, 0)

    # Initialize P
    P = np.zeros((n, n))

    # Initialize betas
    betas = np.ones((n, 1))

    # Shannon entropy (base 2)
    H = np.log2(perplexity)

    return D, P, betas, H
