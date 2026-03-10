#!/usr/bin/env python3
"""Plot exponential decay of C-14 and Ra-226 as line graphs."""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """Plot two exponential decay curves with appropriate labels and legend."""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    plt.figure(figsize=(6.4, 4.8))

    # Plot C-14 (dashed red) and Ra-226 (solid green)
    plt.plot(x, y1, "r--", label="C-14")
    plt.plot(x, y2, "g-", label="Ra-226")

    # Labels and title
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")

    # Set axis ranges
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Legend in upper right
    plt.legend(loc="upper right")
