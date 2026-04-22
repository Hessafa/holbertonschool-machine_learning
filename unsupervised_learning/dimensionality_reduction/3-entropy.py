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

    # Sum of probabilities
    sum_P = np.sum(P)

    # Avoid division by zero
    if sum_P == 0:
        Pi = np.zeros_like(P)
        Hi = 0
        return Hi, Pi

    # Normalize
    Pi = P / sum_P

    # Compute entropy
    Hi = np.log2(sum_P) + beta * np.sum(Di * P) / sum_P

    return Hi, Pi
