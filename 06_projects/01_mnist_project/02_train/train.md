# 🚀 MNIST Project: Train

> 📚 **Projects · Mnist Project** &nbsp;|&nbsp; 🧩 Lesson 2 of 3 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🟣 Project
>
> Progress: `▰▰▱` 2/3

---

## 🎯 Learning Goal
Train the MNIST model and save it.

## 📌 What is it?
This script loads MNIST, trains the CNN for 3 epochs, prints the test accuracy after each epoch, and saves the learned weights to `model.pth`.

## 🧠 Real-Life Analogy
Like practicing a skill for a few rounds, checking your score after each round, and then writing down what you learned.

---

## 💻 Example

Code from `train.py` (run it with `python train.py`):

```python
# Train the model on MNIST digits and save the best version
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
# The model class lives in the sibling folder 01_model
sys.path.insert(0, os.path.join(HERE, "..", "01_model"))
from model import Net

# ----- settings -----
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# ----- data -----
transform = transforms.ToTensor()
data_folder = os.path.join(HERE, "..", "data")
train_data = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=data_folder, train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)
print("train images:", len(train_data), "| test images:", len(test_data))

# ----- model, loss, optimizer -----
model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch():
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test_accuracy():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return 100 * correct / len(test_data)


# ----- training -----
best_accuracy = 0.0
for epoch in range(1, EPOCHS + 1):
    avg_loss = train_one_epoch()
    accuracy = test_accuracy()
    print(f"Epoch {epoch}/{EPOCHS} | loss = {avg_loss:.4f} | test accuracy = {accuracy:.2f}%")
    if accuracy > best_accuracy:                 # keep only the best model
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(HERE, "model.pth"))

print(f"Best accuracy: {best_accuracy:.2f}% (model saved to model.pth)")
```

## 📤 Expected Output

```text
train images: 60000 | test images: 10000
Epoch 1/3 | loss = 0.xxxx | test accuracy = 98.xx%
Epoch 2/3 | loss = 0.xxxx | test accuracy = 98.xx%
Epoch 3/3 | loss = 0.xxxx | test accuracy = 98.xx%
Best accuracy: 98.xx% (model saved to model.pth)

(Roughly 98-99% for MNIST. Needs internet for the first download.)
```

## 🔍 Explanation
1. Settings (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`) are at the top.
2. The model class is imported from `../01_model`.
3. `train_one_epoch()` runs the five-step loop and returns the average loss.
4. `test_accuracy()` measures accuracy on the test set.
5. Each epoch prints loss and accuracy; only the best model is saved to `model.pth`.

---

## 📐 Where Files Go

```
06_projects/<project>/
├── 01_model/model.py
├── 02_train/train.py   -> creates 02_train/model.pth
├── 03_predict/predict.py
└── data/               (downloaded automatically)
```

## 📝 Important Points

- ✅ Run `train.py` before `predict.py`.
- ✅ Accuracy should rise each epoch.
- ✅ Adam is an optimizer that usually works well with little tuning.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Running predict before train | Train first so `model.pth` exists |
| Not keeping the best model | Save when accuracy improves |

## 🧪 Try It Yourself

- [ ] Change `EPOCHS` to 5.
- [ ] Try `LEARNING_RATE = 0.01`.
- [ ] Print the loss every 100 batches.

## ❓ Quick Questions

1. What does `torch.save(model.state_dict(), ...)` save?
2. Why evaluate after each epoch?
3. What does Adam do?

---

## 🏁 Summary

> 💡 Train, test and save the best MNIST model.

⬅️ **Previous:** [Model](../01_model/model.md)  |  ➡️ **Next:** [Predict](../03_predict/predict.md)
