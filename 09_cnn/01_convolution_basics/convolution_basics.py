# Lesson 1: Convolution basics
# A convolution slides a small filter (kernel) over an image.
# At each position: multiply the numbers under the filter and add them up.

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== 1. A tiny image and a filter ===")
image = torch.tensor([[1., 2., 3., 0., 1.],
                      [0., 1., 2., 3., 1.],
                      [1., 0., 1., 2., 2.],
                      [2., 1., 0., 1., 0.],
                      [1., 2., 1., 0., 1.]])
kernel = torch.tensor([[1., 0., -1.],
                       [1., 0., -1.],
                       [1., 0., -1.]])
print("image 5x5, filter 3x3")

print("\n=== 2. Convolution by hand (with loops) ===")
size = image.shape[0] - kernel.shape[0] + 1          # 5 - 3 + 1 = 3
result = torch.zeros(size, size)
for row in range(size):
    for col in range(size):
        patch = image[row:row + 3, col:col + 3]      # the 3x3 area under the filter
        result[row, col] = (patch * kernel).sum()
print(result)

print("\n=== 3. The same with PyTorch (F.conv2d) ===")
# conv2d wants shape (batch, channels, height, width)
fast = F.conv2d(image.reshape(1, 1, 5, 5), kernel.reshape(1, 1, 3, 3))
print(fast[0, 0])
print("same as the loops?", torch.allclose(result, fast[0, 0]))

print("\n=== 4. A vertical edge detector in action ===")
picture = torch.zeros(6, 6)
picture[:, 3:] = 1.0                                  # left half dark, right half bright
for row in picture:
    print(" ".join("#" if p > 0.5 else "." for p in row))
edges = F.conv2d(picture.reshape(1, 1, 6, 6), kernel.reshape(1, 1, 3, 3))[0, 0]
print("filter response (big numbers = edge found):")
print(edges)

print("\n=== 5. Output size formula ===")
print("output = (size - kernel + 2*padding) / stride + 1")
for size_in, k, pad, stride in [(28, 3, 0, 1), (28, 3, 1, 1), (28, 5, 0, 1), (28, 3, 1, 2)]:
    out = (size_in - k + 2 * pad) // stride + 1
    print(f"input {size_in}, kernel {k}, padding {pad}, stride {stride} -> output {out}")

print("\n=== 6. nn.Conv2d learns its own filters ===")
conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
print("weight shape:", tuple(conv.weight.shape), "= (filters, channels, height, width)")
print("parameters:", sum(p.numel() for p in conv.parameters()), "= 4*(1*3*3) + 4 biases")
print("output for a (1, 1, 28, 28) image:", tuple(conv(torch.rand(1, 1, 28, 28)).shape))

print("\n=== 7. Colour images have 3 channels ===")
color_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
print("input (1, 3, 32, 32) -> output", tuple(color_conv(torch.rand(1, 3, 32, 32)).shape))
print("weight shape:", tuple(color_conv.weight.shape))
