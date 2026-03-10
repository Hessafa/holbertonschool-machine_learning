#!/usr/bin/env python3
"""Scatter plot of men's height vs weight."""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """Plot men's height vs weight as a magenta scatter plot."""
    # Mean and covariance for generating data
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)

    # Generate data
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180  # Adjust weight

    # Create figure
    plt.figure(figsize=(6.4, 4.8))

    # Scatter plot
    plt.scatter(x, y, color='m')  # Magenta points

    # Label axes
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')

    # Set title
    plt.title("Men's Height vs Weight")

    # Show plot
    plt.show()
