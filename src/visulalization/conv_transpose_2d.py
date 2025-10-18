import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.randn(1, 1, 8, 8)

deconv = nn.ConvTranspose2d(
    in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False
)

# 出力
y = deconv(x).detach()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Input (8×8)")
plt.imshow(x[0, 0], cmap="magma")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Output (Deconv stride=2)")
plt.imshow(y[0, 0], cmap="magma")
plt.axis("off")

plt.tight_layout()
plt.show()
