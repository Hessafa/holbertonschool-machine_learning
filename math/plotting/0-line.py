#!/usr/bin/env python3
"""Line graph of y = x^3."""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot y = x^3 as a solid red line."""
    # Generate data
    x = np.arange(0, 11)
    y = x ** 3

    # Create figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot line
    plt.plot(x, y, 'r-')  # 'r-' = red solid line

    # Set x-axis limits
    plt.xlim(0, 10)

    # Show the plot
    plt.show()
