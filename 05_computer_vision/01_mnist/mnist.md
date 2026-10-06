# 👁️ Mnist

> 📚 **Computer Vision** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Load the MNIST handwritten-digit dataset.

## 📌 What is it?
**MNIST** has 70,000 grayscale images of handwritten digits (0-9), each 28×28 pixels. It is the 'Hello World' of computer vision.

## 🧠 Real-Life Analogy
It's like a notebook of handwriting practice by thousands of people, each page labeled with the digit it shows.

---

## 💻 Example

Code from `mnist.py` (run it with `python mnist.py`):

```python
# Lesson 1: MNIST
# 70,000 handwritten digits (0-9), each a 28x28 grayscale image.
# The first run downloads the data into a folder called "data" (needs internet).

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ToTensor turns a picture into a tensor with values between 0 and 1
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

print("=== 1. Dataset size ===")
print("Training images:", len(train_data))
print("Test images:", len(test_data))

print("\n=== 2. One image ===")
image, label = train_data[0]
print("One image shape:", image.shape)        # (1, 28, 28) = channels, height, width
print("Its label:", label)
print("Pixel range:", image.min().item(), "to", image.max().item())

print("\n=== 3. Drawing the digit with text ===")
for row in image[0][::2]:                     # every 2nd row so it fits on screen
    print("".join("#" if p > 0.5 else "." for p in row))

print("\n=== 4. How many of each digit? (first 1000 images) ===")
counts = [0] * 10
for i in range(1000):
    counts[train_data[i][1]] += 1
for digit, count in enumerate(counts):
    print(f"digit {digit}: {count:3d} {'#' * (count // 5)}")

print("\n=== 5. Batches ===")
loader = DataLoader(train_data, batch_size=64, shuffle=True)
images, labels = next(iter(loader))
print("Batch of images:", images.shape)       # (64, 1, 28, 28)
print("Batch of labels:", labels.shape)       # (64,)
print("First 10 labels:", labels[:10].tolist())

print("\n=== 6. Average pixel value ===")
print("mean pixel of this batch:", round(images.mean().item(), 3))
```

## 📤 Expected Output

```text
=== 1. Dataset size ===
Training images: 60000
Test images: 10000

=== 2. One image ===
One image shape: torch.Size([1, 28, 28])
Its label: 5
Pixel range: 0.0 to 1.0

=== 3. Drawing the digit with text ===
(a 14-line picture of the digit 5 made of # and .)

=== 4. How many of each digit? (first 1000 images) ===
digit 0:  97 ###################
digit 1: 116 #######################
...
(one line for each digit 0-9)

=== 5. Batches ===
Batch of images: torch.Size([64, 1, 28, 28])
Batch of labels: torch.Size([64])
First 10 labels: [3, 7, 0, 1, 9, 4, 4, 2, 8, 6]

=== 6. Average pixel value ===
mean pixel of this batch: 0.13

(Needs internet for the first download. Batch labels and counts change a little each run.)
```

## 🔍 Explanation
1. Loads MNIST with `ToTensor()` (60,000 train, 10,000 test images).
2. Prints the shape, label and pixel range of one image.
3. Draws the digit with `#` and `.` characters.
4. Counts how many of each digit appear in the first 1000 images.
5. Creates a DataLoader and prints batch shapes.
6. Prints the average pixel value.

---

## 📐 Image Shapes

```
one image  : (1, 28, 28)       channels, height, width
a batch    : (64, 1, 28, 28)   batch, channels, height, width
labels     : (64,)
```

60,000 training images and 10,000 test images.

## 📝 Important Points

- ✅ Pixel values are between 0 (black) and 1 (white).
- ✅ Gray images have 1 channel; color images have 3.
- ✅ The data is saved in the `data` folder, so it's downloaded only once.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting `ToTensor()` | Images would stay PIL pictures |
| Expecting 255 as the max pixel | `ToTensor` scales to 0-1 |

## 🧪 Try It Yourself

- [ ] Print the label of image 10.
- [ ] Count how many 0s are in the first 1000 labels.
- [ ] Print the average pixel value of one image.

## ❓ Quick Questions

1. How big is an MNIST image?
2. What does `ToTensor` do?
3. What shape is a batch of 64 images?

---

## 🏁 Summary

> 💡 MNIST = 28×28 grayscale digits; load it, look at it, batch it.

➡️ **Next:** [Fashion Mnist](../02_fashion_mnist/fashion_mnist.md)
