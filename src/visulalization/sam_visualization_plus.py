import numpy as np
import matplotlib.pyplot as plt


def loss(xy):
    x, y = xy[..., 0], xy[..., 1]
    a, b = 0.2, 1.0
    base = 0.5 * (a * x**2 + b * y**2)

    A1 = 1.6
    x1, y1 = 1.2, -0.8
    s1x, s1y = 0.03, 0.0025
    u1 = (x - x1) ** 2 / s1x + (y - y1) ** 2 / s1y
    sharp = -A1 * np.exp(-u1)

    A2 = 1.2
    x2, y2 = -0.6, 0.6
    s2x, s2y = 0.7, 0.5
    u2 = (x - x2) ** 2 / s2x + (y - y2) ** 2 / s2y
    flat = -A2 * np.exp(-u2)

    return base + sharp + flat


def grad(w):
    x, y = w[0], w[1]
    a, b = 0.2, 1.0
    gx = a * x
    gy = b * y

    A1 = 1.6
    x1, y1 = 1.2, -0.8
    s1x, s1y = 0.03, 0.0025
    u1 = (x - x1) ** 2 / s1x + (y - y1) ** 2 / s1y
    e1 = np.exp(-u1)
    gx += A1 * e1 * (2 * (x - x1) / s1x)
    gy += A1 * e1 * (2 * (y - y1) / s1y)

    A2 = 1.2
    x2, y2 = -0.6, 0.6
    s2x, s2y = 0.7, 0.5
    u2 = (x - x2) ** 2 / s2x + (y - y2) ** 2 / s2y
    e2 = np.exp(-u2)
    gx += A2 * e2 * (2 * (x - x2) / s2x)
    gy += A2 * e2 * (2 * (y - y2) / s2y)

    return np.array([gx, gy], dtype=float)


def step_gd(w, lr):
    return w - lr * grad(w)


def step_sam(w, lr, rho, eps=1e-12):
    g = grad(w)
    gnorm = np.linalg.norm(g) + eps
    w_pert = w + rho * g / gnorm
    g_sam = grad(w_pert)
    return w - lr * g_sam


def simulate(w0, steps=140, lr=0.08, rho=0.25):
    wg = w0.copy()
    ws = w0.copy()
    traj_gd = [wg.copy()]
    traj_sam = [ws.copy()]
    loss_gd = [loss(wg)]
    loss_sam = [loss(ws)]
    sharp_gd = []
    sharp_sam = []
    for _ in range(steps):
        # sharpness proxy before update
        for w, sharp_list in [(wg, sharp_gd), (ws, sharp_sam)]:
            g = grad(w)
            gnorm = np.linalg.norm(g) + 1e-12
            w_pert = w + rho * g / gnorm
            sharp_list.append(loss(w_pert) - loss(w))

        wg = step_gd(wg, lr)
        ws = step_sam(ws, lr, rho)
        traj_gd.append(wg.copy())
        traj_sam.append(ws.copy())
        loss_gd.append(loss(wg))
        loss_sam.append(loss(ws))

    return (
        np.array(traj_gd),
        np.array(traj_sam),
        np.array(loss_gd),
        np.array(loss_sam),
        np.array(sharp_gd),
        np.array(sharp_sam),
    )


def make_all_figures():
    # Grid for contours
    xs = np.linspace(-2.0, 2.0, 300)
    ys = np.linspace(-1.8, 1.8, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = loss(np.stack([X, Y], axis=-1))

    # w0 = np.array([1.6, 1.2])
    w0 = np.array([1.6, -1.0])
    traj_gd, traj_sam, loss_gd, loss_sam, sharp_gd, sharp_sam = simulate(w0)

    # 1) Contour + trajectories
    plt.figure(figsize=(7.5, 6.2))
    levels = np.linspace(Z.min(), min(Z.max(), 3.0), 25)
    plt.contour(X, Y, Z, levels=levels, linewidths=0.8)
    plt.plot(traj_gd[:, 0], traj_gd[:, 1], label="Vanilla GD")
    plt.plot(traj_sam[:, 0], traj_sam[:, 1], label="SAM (ρ=0.25)")
    # Mark putative basin centers for reference
    plt.scatter([1.2, -0.6], [-0.8, 0.6], s=40)
    plt.text(1.25, -0.8, "sharp basin", fontsize=9)
    plt.text(-0.95, 0.65, "flat basin", fontsize=9)
    # Start marker
    plt.scatter([w0[0]], [w0[1]], s=50, marker="*")
    plt.title("SAM vs. Vanilla GD: trajectories on a sharp/flat landscape")
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("sam_contour_trajectories.png", dpi=180)

    # 2) Loss vs. step
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(loss_gd, label="Vanilla GD")
    plt.plot(loss_sam, label="SAM (ρ=0.25)")
    plt.title("Objective vs. step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sam_loss_vs_step.png", dpi=180)

    # 3) Sharpness proxy vs. step
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(sharp_gd, label="Vanilla GD")
    plt.plot(sharp_sam, label="SAM (ρ=0.25)")
    plt.title("Sharpness proxy ΔρL = L(w+ρ g/||g||) - L(w) vs. step")
    plt.xlabel("Step")
    plt.ylabel("ΔρL (bigger = sharper)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sam_sharpness_proxy.png", dpi=180)

    # Show the last figure just to render in notebook UIs
    plt.show()


if __name__ == "__main__":
    make_all_figures()
