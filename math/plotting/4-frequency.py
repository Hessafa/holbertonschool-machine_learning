#!/usr/bin/env python3
"""Plot a histogram of student grades for Project A."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot histogram of student grades with bins of 10 and black edges."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Histogram: bins every 10 units, bars outlined in black
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")

    # Labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Ensure the plot is rendered (required by automated tests)
    plt.show()
