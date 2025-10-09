import numpy as np
import matplotlib.pyplot as plt


def adam_momentum_demo():
    np.random.seed(0)
    steps = 30
    grads = np.sin(np.linspace(0, 4 * np.pi, steps)) + np.random.normal(0, 0.2, steps)
    beta1 = 0.9
    m = np.zeros(steps)
    for t in range(1, steps):
        m[t] = beta1 * m[t - 1] + (1 - beta1) * grads[t]
    plt.figure(figsize=(6, 3))
    plt.plot(grads, label="Gradient $g_t$", color="#ff7f0e", alpha=0.8)
    plt.plot(m, label="Momentum $m_t$", color="#1f77b4", linewidth=2)
    plt.title("Adam First-Moment Intuition (Momentum Effect)")
    plt.xlabel("Step")
    plt.legend()
    plt.tight_layout()
    plt.show()


adam_momentum_demo()
