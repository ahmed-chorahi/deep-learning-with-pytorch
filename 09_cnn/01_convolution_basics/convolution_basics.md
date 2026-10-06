# 🖼️ Convolution Basics: Sliding a Filter

> 📚 **CNN (Convolutional Neural Networks)** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Understand what a convolution does, by doing one by hand.

## 📌 What is it?
A **convolution** slides a small grid of numbers (a **filter** or **kernel**) over an image. At each position it multiplies and adds. Different filters detect different patterns, like edges.

## 🧠 Real-Life Analogy
Like moving a small stencil over a drawing and noting how well the drawing matches the stencil at each spot.

---

## 💻 Example

Code from `convolution_basics.py` (run it with `python convolution_basics.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. A tiny image and a filter ===
image 5x5, filter 3x3

=== 2. Convolution by hand (with loops) ===
tensor([[-4., -2.,  2.],
        [ 0., -4.,  0.],
        [ 2.,  0., -1.]])

=== 3. The same with PyTorch (F.conv2d) ===
tensor([[-4., -2.,  2.],
        [ 0., -4.,  0.],
        [ 2.,  0., -1.]])
same as the loops? True

=== 4. A vertical edge detector in action ===
. . . # # #
. . . # # #
. . . # # #
. . . # # #
. . . # # #
. . . # # #
filter response (big numbers = edge found):
tensor([[ 0., -3., -3.,  0.],
        [ 0., -3., -3.,  0.],
        [ 0., -3., -3.,  0.],
        [ 0., -3., -3.,  0.]])

=== 5. Output size formula ===
output = (size - kernel + 2*padding) / stride + 1
input 28, kernel 3, padding 0, stride 1 -> output 26
input 28, kernel 3, padding 1, stride 1 -> output 28
input 28, kernel 5, padding 0, stride 1 -> output 24
input 28, kernel 3, padding 1, stride 2 -> output 14

=== 6. nn.Conv2d learns its own filters ===
weight shape: (4, 1, 3, 3) = (filters, channels, height, width)
parameters: 40 = 4*(1*3*3) + 4 biases
output for a (1, 1, 28, 28) image: (1, 4, 26, 26)

=== 7. Colour images have 3 channels ===
input (1, 3, 32, 32) -> output (1, 8, 32, 32)
weight shape: (8, 3, 3, 3)

Note: random numbers are used, so your values will differ.
```

## 🔍 Explanation
1. A 5×5 image and a 3×3 vertical-edge filter are created.
2. Loops slide the filter over the image: multiply the 3×3 patch by the filter and add up.
3. `F.conv2d` gives the same answer much faster.
4. A 6×6 picture with a dark left half and bright right half shows the filter finding the edge.
5. The size formula `(size - kernel + 2*padding) / stride + 1` is applied to examples.
6. `nn.Conv2d` learns its own filters; its weight shape is `(filters, channels, h, w)`.
7. Colour images have 3 input channels.

---

## 📐 Sliding a Filter

```
image (5x5)             filter (3x3)         result (3x3)
┌─┬─┬─┬─┬─┐             ┌──┬──┬──┐           ┌─┬─┬─┐
│ │ │ │ │ │   slide     │ 1│ 0│-1│    ==>    │ │ │ │
├─┼─┼─┼─┼─┤   ──────►   ├──┼──┼──┤           ├─┼─┼─┤
│ │ │ │ │ │             │ 1│ 0│-1│           │ │ │ │
└─┴─┴─┴─┴─┘             └──┴──┴──┘           └─┴─┴─┘
```

Input shape for PyTorch: `(batch, channels, height, width)`.

## 📝 Important Points

- ✅ The same filter is used at every position (weights are shared).
- ✅ A big response means the pattern was found.
- ✅ `Conv2d` weights are learned during training.
- ✅ Output size depends on kernel, padding and stride.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Wrong input shape for `conv2d` | Use `(batch, channels, H, W)` |
| Mismatched channels | `in_channels` must equal the input's channels |

## 🧪 Try It Yourself

- [ ] Make a horizontal edge filter (rotate the numbers) and test it.
- [ ] Change the kernel to all ones / 9 (a blur).
- [ ] Compute the output size for 32, kernel 5, padding 2.

## ❓ Quick Questions

1. What does a filter do?
2. What shape has the weight of `Conv2d(3, 8, 3)`?
3. Why is the output smaller than the input without padding?

---

## 🏁 Summary

> 💡 A convolution multiplies and adds a small filter at every position to find patterns.

➡️ **Next:** [Pooling Padding Stride](../02_pooling_padding_stride/pooling_padding_stride.md)
