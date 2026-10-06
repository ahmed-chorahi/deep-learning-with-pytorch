# 🧪 Cnn Questions

> 📚 **Practice** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
Review convolution and pooling shape calculations.

## 📌 What is it?
Six questions about how convolutions and pooling change the size of images.

## 🧠 Real-Life Analogy
Like measuring how a photo shrinks as you crop and zoom it.

---

## 💻 Example

Code from `cnn_questions.py` (run it with `python cnn_questions.py`):

```python
# CNN questions
import torch
import torch.nn as nn


def conv_output_size(size, kernel, padding=0, stride=1):
    # formula: (size - kernel + 2*padding) / stride + 1
    return (size - kernel + 2 * padding) // stride + 1


# Q1: Output shape of this convolution?
conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)   # no padding
x = torch.rand(1, 3, 32, 32)
print("Q1:", tuple(conv(x).shape), "(32 - 3 + 1 = 30)")

# Q2: How does padding=1 change it?
conv_pad = nn.Conv2d(3, 6, kernel_size=3, padding=1)
print("Q2:", tuple(conv_pad(x).shape), "(size stays the same)")

# Q3: What does MaxPool2d(2) do to the size?
print("Q3:", tuple(nn.MaxPool2d(2)(torch.rand(1, 6, 32, 32)).shape), "(halved)")

# Q4: How many numbers go into the Linear layer after flatten?
out = nn.MaxPool2d(2)(conv_pad(x))
print("Q4:", out.flatten(start_dim=1).shape[1], "= 6 * 16 * 16")

# Q5: Complete a CNN for 28x28 grayscale images with 10 classes
model = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> (4, 14, 14)
    nn.Flatten(),
    nn.Linear(4 * 14 * 14, 10),
)
print("Q5:", tuple(model(torch.rand(2, 1, 28, 28)).shape))

# Q6: Why use a CNN instead of only Linear layers for images?
print("Q6: CNN filters are reused across the image, so they need far fewer parameters")
print("    and can find patterns (like edges) anywhere in the picture.")

# Q7: Check our formula against PyTorch
size = 32
print("\nQ7: size after each step (formula)")
size = conv_output_size(size, kernel=3, padding=1)
print("  conv 3x3, padding 1 ->", size)
size = size // 2
print("  pool 2              ->", size)
size = conv_output_size(size, kernel=5)
print("  conv 5x5, no padding->", size)

# Q8: Count parameters of Conv2d(3, 6, 3):  6 * (3*3*3) weights + 6 biases
count = sum(p.numel() for p in conv.parameters())
print("\nQ8:", count, "= 6*(3*3*3) + 6")
```

## 📤 Expected Output

```text
Q1: (1, 6, 30, 30) (32 - 3 + 1 = 30)
Q2: (1, 6, 32, 32) (size stays the same)
Q3: (1, 6, 16, 16) (halved)
Q4: 1536 = 6 * 16 * 16
Q5: (2, 10)
Q6: CNN filters are reused across the image, so they need far fewer parameters
    and can find patterns (like edges) anywhere in the picture.

Q7: size after each step (formula)
  conv 3x3, padding 1 -> 32
  pool 2              -> 16
  conv 5x5, no padding-> 12

Q8: 168 = 6*(3*3*3) + 6

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. A helper `conv_output_size()` applies the size formula.
2. Q1-Q2: effect of padding.
3. Q3-Q4: pooling and flatten size.
4. Q5: complete a CNN for 28×28 images.
5. Q6: why CNNs.
6. Q7: compare the formula with PyTorch step by step.
7. Q8: count Conv2d parameters (168).

---

## 📐 Size Formula

```
output = (input - kernel + 2*padding) / stride + 1
(32 - 3 + 0) / 1 + 1 = 30
(32 - 3 + 2) / 1 + 1 = 32
```

## 📝 Important Points

- ✅ Padding keeps the size; pooling shrinks it.
- ✅ Flatten size = channels × height × width.
- ✅ Compute shapes by hand before running.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting padding in the formula | size − kernel + 2·padding, divided by stride, plus 1 |

## 🧪 Try It Yourself

- [ ] Compute the output for `kernel_size=5, padding=0` on 32×32.
- [ ] Use `MaxPool2d(4)` on 32×32.
- [ ] Make a CNN for 32×32 color images.

## ❓ Quick Questions

1. What does `padding=1` do for a 3×3 kernel?
2. What is the flatten size after Q3?
3. Why use CNNs for images?

---

## 🏁 Summary

> 💡 Eight questions about convolution sizes and CNN design.

⬅️ **Previous:** [Neural Network Questions](../02_neural_network_questions/neural_network_questions.md)  |  ➡️ **Next:** [Pytorch Interview](../04_pytorch_interview/pytorch_interview.md)
