# 👁️ Test Images

> 📚 **Computer Vision** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Measure how accurate a model is on the test set.

## 📌 What is it?
**Accuracy** is the percentage of test images the model gets right. We train quickly, then count correct predictions on 10,000 test images the model has never seen.

## 🧠 Real-Life Analogy
Like grading a quiz: count the right answers and divide by the total number of questions.

---

## 💻 Example

Code from `test_images.py` (run it with `python test_images.py`):

```python
# Lesson 5: Testing a model on the whole test set
# Accuracy = correct predictions / total predictions.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# Same model as in 03_cnn/cnn.py (copied so this lesson runs on its own)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.flatten(start_dim=1))


torch.manual_seed(0)
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(Subset(train_data, range(5000)), batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


print("=== 1. Accuracy BEFORE training (about 10% = random guessing) ===")
print(f"{accuracy(model, test_loader):.2f}%")

print("\n=== 2. Training ===")
for epoch in range(2):
    model.train()
    for images, labels in train_loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"after epoch {epoch + 1}: test accuracy = {accuracy(model, test_loader):.2f}%")

print("\n=== 3. Accuracy for each digit ===")
correct = [0] * 10
total = [0] * 10
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images).argmax(dim=1)
        for label, pred in zip(labels, predictions):
            total[label] += 1
            correct[label] += int(label == pred)
for digit in range(10):
    acc = 100 * correct[digit] / total[digit]
    print(f"digit {digit}: {acc:5.1f}% {'#' * int(acc // 5)}")

print("\n=== 4. Some mistakes ===")
images, labels = next(iter(test_loader))
with torch.no_grad():
    predictions = model(images).argmax(dim=1)
wrong = (predictions != labels).nonzero().flatten()
print("wrong in the first batch of 256:", len(wrong))
for i in wrong[:5]:
    print(f"  true = {labels[i].item()}, predicted = {predictions[i].item()}")
```

## 📤 Expected Output

```text
=== 1. Accuracy BEFORE training (about 10% = random guessing) ===
9.xx%

=== 2. Training ===
after epoch 1: test accuracy = 9x.xx%
after epoch 2: test accuracy = 9x.xx%

=== 3. Accuracy for each digit ===
digit 0:  98.x% ###################
digit 1:  98.x% ###################
...

=== 4. Some mistakes ===
wrong in the first batch of 256: 8
  true = 4, predicted = 9
  ...

(Exact numbers vary. Needs internet for the first MNIST download.)
```

## 🔍 Explanation
1. Defines an `accuracy()` function.
2. Shows accuracy before training (about 10%).
3. Trains for 2 epochs and prints accuracy after each.
4. Calculates the accuracy for each digit with a text bar.
5. Lists a few wrong predictions.

---

## 📐 Accuracy Formula

```
accuracy = correct / total * 100
```

Example: 9,500 right out of 10,000 = 95%.

## 📝 Important Points

- ✅ Always test on data the model has not trained on.
- ✅ Use `model.eval()` and `no_grad`.
- ✅ A simple CNN can reach over 95% on MNIST with a little training.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Testing on training images | Use the test set |
| Computing accuracy with gradients on | Use `torch.no_grad()` |

## 🧪 Try It Yourself

- [ ] Check the accuracy before training (about 10%).
- [ ] Train for 5 epochs and compare.
- [ ] Find which digits are most often wrong.

## ❓ Quick Questions

1. What is accuracy?
2. Why test on unseen data?
3. What does `(predictions == labels).sum()` calculate?

---

## 🏁 Summary

> 💡 Accuracy = correct / total, measured on unseen test data.

⬅️ **Previous:** [Image Prediction](../04_image_prediction/image_prediction.md)
