#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """Plots a histogram of student grades"""

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # bins every 10 units
    bins = np.arange(0, 101, 10)

    # plot histogram
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # display plot
    plt.show()
