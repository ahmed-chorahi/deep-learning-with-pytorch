# Lesson 2: Pooling, padding and stride
# These three settings control how big the image is after each layer.

import torch
import torch.nn as nn

image = torch.tensor([[1., 3., 2., 4.],
                      [5., 6., 1., 2.],
                      [7., 2., 9., 1.],
                      [3., 4., 5., 8.]]).reshape(1, 1, 4, 4)
print("image:\n", image[0, 0])

print("\n=== 1. Max pooling (keep the biggest number in each 2x2 block) ===")
print(nn.MaxPool2d(2)(image)[0, 0])

print("\n=== 2. Average pooling (average of each 2x2 block) ===")
print(nn.AvgPool2d(2)(image)[0, 0])

print("\n=== 3. Global average pooling (one number per channel) ===")
print(nn.AdaptiveAvgPool2d(1)(image)[0, 0])

print("\n=== 4. Padding keeps the image size ===")
x = torch.rand(1, 1, 8, 8)
for padding in [0, 1, 2]:
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=padding)
    print(f"padding = {padding} -> output {tuple(conv(x).shape[2:])}")

print("\n=== 5. Stride = how far the filter jumps ===")
for stride in [1, 2, 3]:
    conv = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=1)
    print(f"stride = {stride} -> output {tuple(conv(x).shape[2:])}")

print("\n=== 6. Following the size through a small CNN ===")
steps = [
    ("input", None),
    ("conv 3x3, padding 1", nn.Conv2d(1, 8, 3, padding=1)),
    ("max pool 2", nn.MaxPool2d(2)),
    ("conv 3x3, padding 1", nn.Conv2d(8, 16, 3, padding=1)),
    ("max pool 2", nn.MaxPool2d(2)),
]
data = torch.rand(1, 1, 28, 28)
for name, layer in steps:
    if layer is not None:
        data = layer(data)
    print(f"{name:22s} -> {tuple(data.shape)}")
print("flatten size:", data.flatten(start_dim=1).shape[1])

print("\n=== 7. Pooling has no parameters ===")
pool = nn.MaxPool2d(2)
print("parameters in MaxPool2d:", sum(p.numel() for p in pool.parameters()))
