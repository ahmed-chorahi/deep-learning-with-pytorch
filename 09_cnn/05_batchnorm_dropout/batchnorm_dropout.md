# 🖼️ BatchNorm and Dropout

> 📚 **CNN (Convolutional Neural Networks)** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Learn two layers that make CNN training more stable and reduce overfitting.

## 📌 What is it?
**BatchNorm** rescales the numbers inside the network so they stay in a healthy range. **Dropout** randomly switches off some neurons during training so the model cannot rely on any single one.

## 🧠 Real-Life Analogy
BatchNorm is like keeping everyone's exam marks on the same scale. Dropout is like a team practicing with random players missing, so everyone learns to play well.

---

## 💻 Example

Code from `batchnorm_dropout.py` (run it with `python batchnorm_dropout.py`):

```python
# Lesson 5: BatchNorm and Dropout
# Two layers that help a CNN train faster and overfit less.

import torch
import torch.nn as nn

torch.manual_seed(0)

print("=== 1. BatchNorm: keeps numbers in a healthy range ===")
x = torch.rand(16, 4, 8, 8) * 10 + 5             # numbers around 5 to 15
print(f"before: mean = {x.mean().item():.2f}, std = {x.std().item():.2f}")
bn = nn.BatchNorm2d(4)
bn.train()
y = bn(x)
print(f"after : mean = {y.mean().item():.2f}, std = {y.std().item():.2f}")

print("\n=== 2. Dropout: randomly switches neurons off while training ===")
drop = nn.Dropout(p=0.5)
data = torch.ones(10)
drop.train()
print("train mode:", drop(data))                  # about half are 0, others scaled to 2
drop.eval()
print("eval mode :", drop(data))                  # nothing is dropped

print("\n=== 3. A CNN with both layers ===")
model_with = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Dropout(0.3),
    nn.Linear(16 * 3 * 3, 3),
)
print(model_with)
print("output shape:", tuple(model_with(torch.rand(2, 1, 12, 12)).shape))

print("\n=== 4. Compare with and without (same data, same epochs) ===")
SIZE = 12


def make_dataset(n):
    labels = torch.randint(0, 3, (n,))
    images = torch.zeros(n, 1, SIZE, SIZE)
    for i, label in enumerate(labels):
        pos = torch.randint(2, SIZE - 2, (1,)).item()
        if label == 0:
            images[i, 0, pos, :] = 1
        elif label == 1:
            images[i, 0, :, pos] = 1
        else:
            for k in range(SIZE):
                images[i, 0, k, k] = 1
    return images + 0.4 * torch.rand_like(images), labels


x_train, y_train = make_dataset(300)
x_test, y_test = make_dataset(150)


def train_and_test(model, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), 50):
            idx = order[start:start + 50]
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_acc = (model(x_train).argmax(dim=1) == y_train).float().mean().item() * 100
        test_acc = (model(x_test).argmax(dim=1) == y_test).float().mean().item() * 100
    print(f"{name:22s} train accuracy = {train_acc:5.1f}% | test accuracy = {test_acc:5.1f}%")


plain = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 3 * 3, 3),
)
fancy = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Dropout(0.3), nn.Linear(16 * 3 * 3, 3),
)
train_and_test(plain, "plain CNN")
train_and_test(fancy, "CNN + BatchNorm + Dropout")
```

## 📤 Expected Output

```text
=== 1. BatchNorm: keeps numbers in a healthy range ===
before: mean = 9.95, std = 2.87
after : mean = -0.00, std = 1.00

=== 2. Dropout: randomly switches neurons off while training ===
train mode: tensor([0., 2., 2., 0., 0., 0., 2., 2., 0., 2.])
eval mode : tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

=== 3. A CNN with both layers ===
Sequential(
  (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
  (2): ReLU()
  (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (4): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
  (6): ReLU()
  (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (8): Flatten(start_dim=1, end_dim=-1)
  (9): Dropout(p=0.3, inplace=False)
  (10): Linear(in_features=144, out_features=3, bias=True)
)
output shape: (2, 3)

=== 4. Compare with and without (same data, same epochs) ===
plain CNN              train accuracy = 100.0% | test accuracy = 100.0%
CNN + BatchNorm + Dropout train accuracy = 100.0% | test accuracy = 100.0%
```

## 🔍 Explanation
1. Part 1: `BatchNorm2d` turns numbers with a large mean into mean ≈ 0 and std ≈ 1.
2. Part 2: `Dropout(0.5)` zeroes about half the values in train mode and does nothing in eval mode.
3. Part 3: a CNN uses both layers; shapes stay the same.
4. Part 4: a plain CNN and a CNN with both layers are trained on the same data and compared.

---

## 📐 Where the Layers Go

```
Conv ─► BatchNorm ─► ReLU ─► Pool        (after convolutions)

Flatten ─► Dropout ─► Linear             (before the final layer)
```

| Layer | Train mode | Eval mode |
|-------|-----------|-----------|
| BatchNorm | uses batch statistics | uses saved averages |
| Dropout | drops neurons | does nothing |

## 📝 Important Points

- ✅ Always call `model.eval()` before testing so Dropout switches off.
- ✅ BatchNorm helps training run faster and more smoothly.
- ✅ Dropout fights overfitting.
- ✅ On this very easy task both models reach 100%; the layers matter more on harder data.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting `model.eval()` | Dropout would still change the results |
| Dropout of 0.9 | Too much; use 0.2 - 0.5 |

## 🧪 Try It Yourself

- [ ] Change dropout to 0.1 and 0.7.
- [ ] Remove BatchNorm and compare the loss.
- [ ] Make the images noisier to see differences.

## ❓ Quick Questions

1. What does BatchNorm do?
2. Why is Dropout switched off at test time?
3. Where do we usually put Dropout?

---

## 🏁 Summary

> 💡 BatchNorm stabilizes training and Dropout reduces overfitting; remember `eval()`.

⬅️ **Previous:** [Cnn Training](../04_cnn_training/cnn_training.md)
