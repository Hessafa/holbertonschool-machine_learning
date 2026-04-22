#!/usr/bin/env python3
"""
Function that calculates Shannon entropy and P affinities.
"""

import numpy as np


def HP(Di, beta):
    """
    Calculates the entropy and P affinities for a data point.

    Args:
        Di (numpy.ndarray): shape (n - 1,)
        beta (numpy.ndarray): shape (1,)

    Returns:
        Hi (float): Shannon entropy
        Pi (numpy.ndarray): P affinities
    """
    # Compute unnormalized probabilities
    P = np.exp(-Di * beta)

    sum_P = np.sum(P)

    # Avoid division by zero
    if sum_P == 0:
        return 0, np.zeros_like(P)

    # Normalize
    Pi = P / sum_P

    # ✅ Correct Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi + 1e-10))

    return Hi, Pi

    # Normalize
    Pi = P / sum_P

    # Compute entropy
    Hi = np.log2(sum_P) + beta * np.sum(Di * P) / sum_P

    return Hi, Pi
