import numpy as np
import matplotlib.pyplot as plt


# SAM visualization
def sam_visual_demo():
    x = np.linspace(-3, 3, 200)
    sharp = x**2
    flat = 0.2 * x**2
    plt.figure(figsize=(6, 3))
    plt.plot(x, sharp, label="Sharp Minimum", color="#ff7f0e")
    plt.plot(x, flat, label="Flat Minimum", color="#1f77b4")
    plt.scatter([0], [0], color="black", zorder=5, s=40)
    plt.title("SAM: Sharp vs. Flat Minima")
    plt.xlabel("Parameter w")
    plt.ylabel("Loss E(w)")
    plt.legend()
    plt.tight_layout()
    plt.show()


sam_visual_demo()
