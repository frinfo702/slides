import numpy as np
import matplotlib.pyplot as plt


# AdaGrad / RMSProp / Adam comparison
def adagrad_rmsprop_adam_demo():
    def f(x, y):
        return 0.5 * x**2 + 2 * y**2

    def grad(x, y):
        return np.array([x, 4 * y])

    methods = ["AdaGrad", "RMSProp", "Adam"]
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c"]
    steps = 50
    lr, eps, gamma = 0.2, 1e-8, 0.9
    beta1, beta2 = 0.9, 0.999
    paths = {m: [] for m in methods}

    for method in methods:
        x, y, Gx, Gy, mx, my, vx, vy = 2.0, 2.0, 0, 0, 0, 0, 0, 0
        for t in range(1, steps + 1):
            g = grad(x, y)
            if method == "AdaGrad":
                Gx += g[0] ** 2
                Gy += g[1] ** 2
                x -= lr / np.sqrt(Gx + eps) * g[0]
                y -= lr / np.sqrt(Gy + eps) * g[1]
            elif method == "RMSProp":
                Gx = gamma * Gx + (1 - gamma) * g[0] ** 2
                Gy = gamma * Gy + (1 - gamma) * g[1] ** 2
                x -= lr / np.sqrt(Gx + eps) * g[0]
                y -= lr / np.sqrt(Gy + eps) * g[1]
            else:  # Adam
                mx = beta1 * mx + (1 - beta1) * g[0]
                my = beta1 * my + (1 - beta1) * g[1]
                vx = beta2 * vx + (1 - beta2) * g[0] ** 2
                vy = beta2 * vy + (1 - beta2) * g[1] ** 2
                mx_hat = mx / (1 - beta1**t)
                my_hat = my / (1 - beta1**t)
                vx_hat = vx / (1 - beta2**t)
                vy_hat = vy / (1 - beta2**t)
                x -= lr * mx_hat / (np.sqrt(vx_hat) + eps)
                y -= lr * my_hat / (np.sqrt(vy_hat) + eps)
            paths[method].append((x, y))

    X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    Z = f(X, Y)
    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, Z, levels=20, cmap="gray", alpha=0.6)
    for method, c in zip(methods, colors):
        p = np.array(paths[method])
        plt.plot(p[:, 0], p[:, 1], marker="o", markersize=2, color=c, label=method)
    plt.title("AdaGrad / RMSProp / Adam Comparison")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


adagrad_rmsprop_adam_demo()
