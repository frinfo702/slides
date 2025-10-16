import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 入力特徴マップ
x = torch.randn(1, 1, 16, 16)

# ConvTranspose (典型的なCheckerboard発生例)
deconv = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)
out_deconv = deconv(x).detach()

# Resize + Conv (アーティファクト少ない)
upsampled = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
out_resizeconv = conv(upsampled).detach()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(out_deconv[0, 0], cmap="magma")
plt.title("ConvTranspose2d (Checkerboard)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(out_resizeconv[0, 0], cmap="magma")
plt.title("Resize + Conv (Smooth)")
plt.axis("off")

plt.tight_layout()
plt.show()
