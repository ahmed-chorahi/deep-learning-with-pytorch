# 👁️ Cnn

> 📚 **Computer Vision** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
Build a simple Convolutional Neural Network (CNN).

## 📌 What is it?
A **CNN** is a network made for images. Small filters slide over the image looking for patterns such as edges and curves, and pooling layers shrink the image while keeping the important parts.

## 🧠 Real-Life Analogy
Like looking at a picture through a small magnifying glass moving across it, noting 'edge here, curve there' and then summarizing.

---

## 💻 Example

Code from `cnn.py` (run it with `python cnn.py`):

```python
# Lesson 3: A simple CNN (Convolutional Neural Network)
# Small filters slide over the image to find patterns like edges and curves.

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)    # (1,28,28) -> (8,28,28)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)   # (8,14,14) -> (16,14,14)
        self.pool = nn.MaxPool2d(2)                               # halves height and width
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 7 * 7, 10)                       # 10 classes

    def forward(self, x, show_shapes=False):
        x = self.pool(self.relu(self.conv1(x)))
        if show_shapes:
            print("after conv1 + pool:", tuple(x.shape))
        x = self.pool(self.relu(self.conv2(x)))
        if show_shapes:
            print("after conv2 + pool:", tuple(x.shape))
        x = x.flatten(start_dim=1)
        if show_shapes:
            print("after flatten     :", tuple(x.shape))
        return self.fc(x)


if __name__ == "__main__":
    model = SimpleCNN()
    print("=== 1. The model ===")
    print(model)

    print("\n=== 2. Shapes through the network ===")
    fake_images = torch.rand(4, 1, 28, 28)         # 4 random "images"
    print("input             :", tuple(fake_images.shape))
    output = model(fake_images, show_shapes=True)
    print("output            :", tuple(output.shape))

    print("\n=== 3. Parameters ===")
    for name, p in model.named_parameters():
        print(f"{name:14s} {str(tuple(p.shape)):18s} {p.numel()}")
    cnn_total = sum(p.numel() for p in model.parameters())
    print("total:", cnn_total)

    print("\n=== 4. CNN vs plain linear layers ===")
    plain = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    plain_total = sum(p.numel() for p in plain.parameters())
    print("plain network parameters:", plain_total)
    print("our CNN parameters      :", cnn_total)

    print("\n=== 5. One filter up close ===")
    print("conv1 weight shape:", tuple(model.conv1.weight.shape), "= (filters, channels, height, width)")
    print("first filter:\n", model.conv1.weight[0, 0].detach().round(decimals=2))
```

## 📤 Expected Output

```text
=== 1. The model ===
SimpleCNN(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu): ReLU()
  (fc): Linear(in_features=784, out_features=10, bias=True)
)

=== 2. Shapes through the network ===
input             : (4, 1, 28, 28)
after conv1 + pool: (4, 8, 14, 14)
after conv2 + pool: (4, 16, 7, 7)
after flatten     : (4, 784)
output            : (4, 10)

=== 3. Parameters ===
conv1.weight   (8, 1, 3, 3)       72
conv1.bias     (8,)               8
conv2.weight   (16, 8, 3, 3)      1152
conv2.bias     (16,)              16
fc.weight      (10, 784)          7840
fc.bias        (10,)              10
total: 9098

=== 4. CNN vs plain linear layers ===
plain network parameters: 101770
our CNN parameters      : 9098

=== 5. One filter up close ===
conv1 weight shape: (8, 1, 3, 3) = (filters, channels, height, width)
first filter:
 tensor([[-0.3200,  0.0200, -0.2000],
        [-0.0100, -0.1300,  0.0100],
        [ 0.2000,  0.2400,  0.1400]])

Note: This lesson uses random numbers, so your values will be different. The filter values are random, so yours will look different.
```

## 🔍 Explanation
1. `SimpleCNN` has two convolution layers, pooling, flatten and a linear output.
2. `show_shapes=True` prints the shape after each stage.
3. A loop lists every parameter and the total.
4. Compares with a plain linear network of the same task.
5. Prints the numbers inside the first 3×3 filter.

---

## 📐 Shapes Through the CNN

```
input      (N, 1, 28, 28)
conv1+pool (N, 8, 14, 14)
conv2+pool (N, 16, 7, 7)
flatten    (N, 784)
linear     (N, 10)
```

## 📝 Important Points

- ✅ `padding=1` with a 3×3 filter keeps the image size the same.
- ✅ Pooling halves the size.
- ✅ CNNs need far fewer parameters than fully-connected layers on images.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Wrong number in the final Linear layer | Work it out: channels × height × width |
| Forgetting `padding=1` | The image would shrink |

## 🧪 Try It Yourself

- [ ] Change 8 to 16 filters in `conv1` (remember to fix `conv2`).
- [ ] Remove one pooling layer and fix the `Linear` size.
- [ ] Count the parameters.

## ❓ Quick Questions

1. What does a convolution do?
2. What does `MaxPool2d(2)` do to the size?
3. Why flatten before the Linear layer?

---

## 🏁 Summary

> 💡 CNNs find patterns with small shared filters and shrink the image with pooling.

⬅️ **Previous:** [Fashion Mnist](../02_fashion_mnist/fashion_mnist.md)  |  ➡️ **Next:** [Image Prediction](../04_image_prediction/image_prediction.md)
