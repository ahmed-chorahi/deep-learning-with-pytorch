# 👁️ Fashion Mnist

> 📚 **Computer Vision** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Load the Fashion-MNIST clothing dataset.

## 📌 What is it?
**Fashion-MNIST** has the same format as MNIST (28×28 grayscale, 10 classes), but the pictures show clothing. It is a slightly harder dataset for practice.

## 🧠 Real-Life Analogy
Same notebook as MNIST, but instead of digits each page shows a shirt, shoe or bag.

---

## 💻 Example

Code from `fashion_mnist.py` (run it with `python fashion_mnist.py`):

```python
# Lesson 2: Fashion-MNIST
# Same format as MNIST (28x28 grayscale, 10 classes), but the pictures are clothes.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print("=== 1. Dataset size ===")
print("Training images:", len(train_data))
print("Test images:", len(test_data))

print("\n=== 2. One image ===")
image, label = train_data[0]
print("Image shape:", image.shape)
print("Label:", label, "->", class_names[label])

print("\n=== 3. Draw it with text ===")
for row in image[0][::2]:
    print("".join("#" if p > 0.5 else "." for p in row))

print("\n=== 4. Class names ===")
for number, name in enumerate(class_names):
    print(f"{number} = {name}")

print("\n=== 5. A batch ===")
loader = DataLoader(train_data, batch_size=32, shuffle=True)
images, labels = next(iter(loader))
print("Batch shape:", images.shape)
print("First 8 items:", [class_names[i] for i in labels[:8]])

print("\n=== 6. Count each class in this batch ===")
for number, name in enumerate(class_names):
    count = (labels == number).sum().item()
    print(f"{name:12s} {count:2d} {'#' * count}")
```

## 📤 Expected Output

```text
=== 1. Dataset size ===
Training images: 60000
Test images: 10000

=== 2. One image ===
Image shape: torch.Size([1, 28, 28])
Label: 9 -> Ankle boot

=== 3. Draw it with text ===
(a 14-line picture of an ankle boot made of # and .)

=== 4. Class names ===
0 = T-shirt/top
1 = Trouser
...
9 = Ankle boot

=== 5. A batch ===
Batch shape: torch.Size([32, 1, 28, 28])
First 8 items: ['Sneaker', 'Coat', 'Bag', 'Dress', 'Shirt', 'Trouser', 'Sandal', 'Pullover']

=== 6. Count each class in this batch ===
T-shirt/top   4 ####
Trouser       3 ###
...

(Needs internet for the first download. The batch contents are random.)
```

## 🔍 Explanation
1. Loads FashionMNIST, same format as MNIST.
2. `class_names` converts label numbers to names.
3. Prints and draws the first image (an ankle boot).
4. Prints all 10 class names.
5. Counts each class inside one batch with a text bar chart.

---

## 📐 The 10 Classes

| 0 T-shirt/top | 1 Trouser | 2 Pullover | 3 Dress | 4 Coat |
|---|---|---|---|---|
| **5 Sandal** | **6 Shirt** | **7 Sneaker** | **8 Bag** | **9 Ankle boot** |

Shapes are the same as MNIST: `(1, 28, 28)` per image.

## 📝 Important Points

- ✅ The code is almost identical to MNIST.
- ✅ Labels are numbers; use a list to turn them into names.
- ✅ Shirts and T-shirts are easily confused, so accuracy is lower than MNIST.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Printing label numbers only | Use `class_names[label]` |
| Assuming it is as easy as MNIST | Shirts/T-shirts are often confused |

## 🧪 Try It Yourself

- [ ] Print the name of the label for image 5.
- [ ] Load only the test set.
- [ ] Count images of each class in a batch.

## ❓ Quick Questions

1. How is Fashion-MNIST like MNIST?
2. How do you turn a label number into a name?
3. How many classes are there?

---

## 🏁 Summary

> 💡 Same code as MNIST, harder pictures.

⬅️ **Previous:** [Mnist](../01_mnist/mnist.md)  |  ➡️ **Next:** [Cnn](../03_cnn/cnn.md)
