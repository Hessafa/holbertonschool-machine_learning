#!/usr/bin/env python3
"""Plot a stacked bar graph for fruit quantities per person."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Create a stacked bar chart of fruit quantities for Farrah, Fred, and Felicia."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8), dpi=100)

    people = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    labels = ["Apples", "Bananas", "Oranges", "Peaches"]

    bottom = np.zeros(3)
    for i in range(fruit.shape[0]):
        plt.bar(
            people,
            fruit[i],
            bottom=bottom,
            color=colors[i],
            width=0.5,
            label=labels[i]
        )
        bottom += fruit[i]

    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
