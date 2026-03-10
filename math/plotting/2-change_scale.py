#!/usr/bin/env python3
"""Line plot of C-14 exponential decay with logarithmic scale."""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Plot exponential decay of C-14 with a logarithmic y-axis."""
    # Time points in years
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    # Create figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot line
    plt.plot(x, y, color='b')  # Blue line

    # Set x and y labels
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')

    # Set title
    plt.title('Exponential Decay of C-14')

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set x-axis limits
    plt.xlim(0, 28650)

    # Show plot
    plt.show()
