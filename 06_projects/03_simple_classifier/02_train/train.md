# 🚀 Simple Classifier: Train

> 📚 **Projects · Simple Classifier** &nbsp;|&nbsp; 🧩 Lesson 2 of 3 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟣 Project
>
> Progress: `▰▰▱` 2/3

---

## 🎯 Learning Goal
Train the circle classifier on data we create ourselves.

## 📌 What is it?
We generate random points between -2 and 2 and label each one 1 if it is inside a circle of radius 1, otherwise 0. The model learns this rule. No downloads are needed.

## 🧠 Real-Life Analogy
Like teaching a child to tell if a dart landed inside the bullseye after showing hundreds of examples.

---

## 💻 Example

Code from `train.py` (run it with `python train.py`):

```python
# Task: is a 2D point inside a circle of radius 1 (class 1) or outside (class 0)?
# We create our own data, so nothing needs to be downloaded.
import os
import sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
# The model class lives in the sibling folder 01_model
sys.path.insert(0, os.path.join(HERE, "..", "01_model"))
from model import Classifier

torch.manual_seed(42)


def make_data(n):
    points = torch.rand(n, 2) * 4 - 2                       # random points in [-2, 2]
    labels = ((points ** 2).sum(dim=1) < 1).long()          # inside the circle?
    return points, labels


# ----- data: train / validation / test -----
x_train, y_train = make_data(2000)
x_val, y_val = make_data(500)
x_test, y_test = make_data(500)
print("points inside the circle in training data:", y_train.sum().item(), "of", len(y_train))


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item() * 100


# ----- model, loss, optimizer -----
model = Classifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----- training -----
best_val = 0.0
for epoch in range(1, 201):
    model.train()
    loss = loss_fn(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_acc = accuracy(model, x_val, y_val)
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(HERE, "model.pth"))

    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | loss = {loss.item():.4f} | validation accuracy = {val_acc:.1f}%")

# ----- final test with the best saved model -----
model.load_state_dict(torch.load(os.path.join(HERE, "model.pth")))
print(f"Test accuracy: {accuracy(model, x_test, y_test):.1f}%")
print("Best model saved to model.pth")
```

## 📤 Expected Output

```text
points inside the circle in training data: 405 of 2000
Epoch  40 | loss = 0.2583 | validation accuracy = 82.8%
Epoch  80 | loss = 0.0685 | validation accuracy = 98.2%
Epoch 120 | loss = 0.0273 | validation accuracy = 99.2%
Epoch 160 | loss = 0.0159 | validation accuracy = 99.4%
Epoch 200 | loss = 0.0111 | validation accuracy = 99.6%
Test accuracy: 99.4%
Best model saved to model.pth
```

## 🔍 Explanation
1. `make_data` creates random points and labels (inside = `x² + y² < 1`).
2. Creates train, validation and test sets.
3. Trains for 200 epochs with Adam.
4. Saves the model whenever validation accuracy improves.
5. Reloads the best model and prints the test accuracy.

---

## 📐 The Rule We Teach

```
inside the circle  <=>  x² + y² < 1
```

Data shapes: points `(2000, 2)`, labels `(2000,)`.

## 📝 Important Points

- ✅ Making your own data is a great way to experiment.
- ✅ The whole dataset fits in memory, so we skip DataLoader here.
- ✅ Results are reproducible thanks to `torch.manual_seed`.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Using test data for choosing the best model | Use validation data for that |
| Forgetting the seed | Results change every run |

## 🧪 Try It Yourself

- [ ] Change the radius to 1.5.
- [ ] Use fewer hidden units and see what happens.
- [ ] Train for 50 epochs only.

## ❓ Quick Questions

1. How are labels created?
2. Why do we use test points?
3. Why set a manual seed?

---

## 🏁 Summary

> 💡 Make your own data, train, validate and test.

⬅️ **Previous:** [Model](../01_model/model.md)  |  ➡️ **Next:** [Predict](../03_predict/predict.md)
