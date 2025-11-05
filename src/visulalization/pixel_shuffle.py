import torch
import torch.nn as nn
import matplotlib.pyplot as plt

r = 2  # 倍率
H, W = 4, 4

# 入力を「チャネルに埋めた特徴」として可視化しやすく
x = torch.arange(1, (r**2) * H * W + 1, dtype=torch.float32).reshape(1, r**2, H, W)
pixelshuffle = nn.PixelShuffle(upscale_factor=r)
y = pixelshuffle(x)


def show_tensor(t, title):
    plt.imshow(t[0, 0], cmap="viridis")
    plt.title(title)
    plt.axis("off")


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
show_tensor(x, f"Input: {r**2}ch × {H}×{W}")

plt.subplot(1, 2, 2)
show_tensor(y, f"Output: 1ch × {H * r}×{W * r}")

plt.tight_layout()
plt.show()
