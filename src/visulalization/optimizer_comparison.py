import numpy as np
import matplotlib.pyplot as plt


def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    # Ill-conditioned quadratic: rotate and scale
    Q = np.array([[10.0, 0.0], [0.0, 0.1]])  # eigenvalues differ by 100x
    theta = np.deg2rad(35)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = R @ Q @ R.T  # SPD matrix
    b = np.array([-2.0, 3.0])

    def f(w):
        return 0.5 * w.T @ A @ w + b.T @ w

    def true_grad(w):
        return A @ w + b

    # Additive stochastic noise to mimic minibatch gradients
    def noisy_grad(w, rng):
        noise = rng.normal(scale=0.5, size=w.shape)  # tune noise for clarity
        return true_grad(w) + noise

    return f, noisy_grad


def adagrad_step(state, grad, lr=0.5, eps=1e-8):
    G = state.get("G", np.zeros_like(grad))
    G += grad * grad
    step = lr / (np.sqrt(G) + eps) * grad
    state["G"] = G
    return -step, state


def rmsprop_step(state, grad, lr=0.05, beta=0.9, eps=1e-8):
    v = state.get("v", np.zeros_like(grad))
    v = beta * v + (1 - beta) * (grad * grad)
    step = lr / (np.sqrt(v) + eps) * grad
    state["v"] = v
    return -step, state


def adam_step(state, grad, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    m = state.get("m", np.zeros_like(grad))
    v = state.get("v", np.zeros_like(grad))
    t = state.get("t", 0) + 1
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * (grad * grad)
    mhat = m / (1 - b1**t)
    vhat = v / (1 - b2**t)
    step = lr * mhat / (np.sqrt(vhat) + eps)
    state.update({"m": m, "v": v, "t": t})
    return -step, state


def sgd_step(state, grad, lr=0.05):
    return -lr * grad, state


def run_optimizer(step_fn, name, f, noisy_grad, w0, steps=200, seed=0):
    rng = np.random.default_rng(seed)
    w = w0.copy()
    hist = []
    state = {}
    for _ in range(steps):
        g = noisy_grad(w, rng)
        delta, state = step_fn(state, g)
        w = w + delta
        hist.append(f(w))
    return name, np.array(hist)


def main():
    f, noisy_grad = make_problem(seed=42)
    w0 = np.array([6.0, -6.0])  # same start for all
    steps = 250

    runs = []
    runs.append(
        run_optimizer(
            lambda s, g: sgd_step(s, g, lr=0.05),
            "SGD (lr=0.05)",
            f,
            noisy_grad,
            w0,
            steps,
            seed=0,
        )
    )
    runs.append(
        run_optimizer(
            lambda s, g: adagrad_step(s, g, lr=0.8),
            "AdaGrad (lr=0.8)",
            f,
            noisy_grad,
            w0,
            steps,
            seed=1,
        )
    )
    runs.append(
        run_optimizer(
            lambda s, g: rmsprop_step(s, g, lr=0.08, beta=0.9),
            "RMSProp (lr=0.08, β=0.9)",
            f,
            noisy_grad,
            w0,
            steps,
            seed=2,
        )
    )
    runs.append(
        run_optimizer(
            lambda s, g: adam_step(s, g, lr=0.2, b1=0.9, b2=0.999),
            "Adam (lr=0.2)",
            f,
            noisy_grad,
            w0,
            steps,
            seed=3,
        )
    )

    plt.figure(figsize=(8, 5))
    for name, hist in runs:
        # Smooth for readability (EWMA)
        alpha = 0.1
        ewma = np.zeros_like(hist)
        acc = 0.0
        for i, v in enumerate(hist):
            acc = alpha * v + (1 - alpha) * (acc if i > 0 else v)
            ewma[i] = acc
        plt.plot(ewma, label=name)
    plt.title(
        "AdaGrad / RMSProp / Adam: noisy ill-conditioned quadratic (loss vs. step)"
    )
    plt.xlabel("Step")
    plt.ylabel("Objective (smoothed)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
