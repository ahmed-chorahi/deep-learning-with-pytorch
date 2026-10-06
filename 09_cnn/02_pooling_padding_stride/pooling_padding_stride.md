# 🖼️ Pooling, Padding and Stride

> 📚 **CNN (Convolutional Neural Networks)** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Learn the three settings that control the size of feature maps.

## 📌 What is it?
**Padding** adds a border so the image does not shrink. **Stride** is how far the filter jumps each step. **Pooling** shrinks the image by keeping one number per block (max or average).

## 🧠 Real-Life Analogy
Padding is a picture frame, stride is the size of your steps when you walk across the picture, and pooling is zooming out so you only keep the highlights.

---

## 💻 Example

Code from `pooling_padding_stride.py` (run it with `python pooling_padding_stride.py`):

```python
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
```

## 📤 Expected Output

```text
image:
 tensor([[1., 3., 2., 4.],
        [5., 6., 1., 2.],
        [7., 2., 9., 1.],
        [3., 4., 5., 8.]])

=== 1. Max pooling (keep the biggest number in each 2x2 block) ===
tensor([[6., 4.],
        [7., 9.]])

=== 2. Average pooling (average of each 2x2 block) ===
tensor([[3.7500, 2.2500],
        [4.0000, 5.7500]])

=== 3. Global average pooling (one number per channel) ===
tensor([[3.9375]])

=== 4. Padding keeps the image size ===
padding = 0 -> output (6, 6)
padding = 1 -> output (8, 8)
padding = 2 -> output (10, 10)

=== 5. Stride = how far the filter jumps ===
stride = 1 -> output (8, 8)
stride = 2 -> output (4, 4)
stride = 3 -> output (3, 3)

=== 6. Following the size through a small CNN ===
input                  -> (1, 1, 28, 28)
conv 3x3, padding 1    -> (1, 8, 28, 28)
max pool 2             -> (1, 8, 14, 14)
conv 3x3, padding 1    -> (1, 16, 14, 14)
max pool 2             -> (1, 16, 7, 7)
flatten size: 784

=== 7. Pooling has no parameters ===
parameters in MaxPool2d: 0

Note: random numbers are used, so your values will differ.
```

## 🔍 Explanation
1. A 4×4 image is pooled with `MaxPool2d(2)` and `AvgPool2d(2)`.
2. `AdaptiveAvgPool2d(1)` gives one number for the whole image.
3. Padding 0, 1, 2 shows how the output size changes.
4. Stride 1, 2, 3 shows how larger jumps shrink the output.
5. A step-by-step list shows the shape after each layer of a small CNN.
6. Pooling layers have zero parameters.

---

## 📐 Max Pooling Example

```
 1  3 | 2  4          max pool 2x2
 5  6 | 1  2    ──►    6  4
-------+------         7  9
 7  2 | 9  1
 3  4 | 5  8
```

| Setting | Effect on size |
|---------|----------------|
| padding 1 (3×3 filter) | stays the same |
| stride 2 | roughly halves |
| pool 2 | exactly halves |

## 📝 Important Points

- ✅ Max pooling keeps the strongest signal.
- ✅ Pooling makes the model less sensitive to small shifts.
- ✅ Padding keeps edge pixels useful.
- ✅ Always track the size to know the flatten size.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Odd image sizes with pool 2 | The last row/column is dropped |
| Wrong Linear size after flatten | Calculate it or test with a dummy tensor |

## 🧪 Try It Yourself

- [ ] Try `MaxPool2d(3)` on a 6×6 image.
- [ ] Use stride 2 in a convolution and print the size.
- [ ] Predict the shape before running.

## ❓ Quick Questions

1. What does max pooling do?
2. What is stride?
3. Why does padding help?

---

## 🏁 Summary

> 💡 Padding, stride and pooling decide how big the image is at every layer.

⬅️ **Previous:** [Convolution Basics](../01_convolution_basics/convolution_basics.md)  |  ➡️ **Next:** [Building Cnn](../03_building_cnn/building_cnn.md)
