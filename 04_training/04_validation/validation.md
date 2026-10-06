# 🏋️ Validation

> 📚 **Training** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Check a model on data it has not seen.

## 📌 What is it?
**Validation** data is held back and never used for learning. Checking the loss on it tells you whether the model really learned or just memorized.

## 🧠 Real-Life Analogy
Like a student practicing with homework (training data) and then taking a surprise quiz (validation data).

---

## 💻 Example

Code from `validation.py` (run it with `python validation.py`):

```python
# Lesson 4: Validation
# We keep some data hidden from training and use it to check the model honestly.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# Same dataset as in 01_dataset/dataset.py (copied so this lesson runs on its own)
class NumbersDataset(Dataset):
    def __init__(self, n=10, noise=0.0):
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1) / 10     # keep numbers small
        self.y = 2 * self.x + 1 + noise * torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


torch.manual_seed(0)

# Split: 40 samples to train on, 10 samples to validate with
full_data = NumbersDataset(n=50, noise=0.2)
train_data, val_data = random_split(full_data, [40, 10])
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

best_val_loss = float("inf")
best_epoch = 0

print("epoch | train loss | val loss")
for epoch in range(1, 101):
    # ----- training -----
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ----- validation (no learning here) -----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            val_loss += loss_fn(model(x), y).item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:          # remember the best epoch
        best_val_loss = val_loss
        best_epoch = epoch

    if epoch % 20 == 0 or epoch == 1:
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f}")

print()
print(f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch}")
print("Learned weight: %.2f (true 2), bias: %.2f (true 1)" % (model.weight.item(), model.bias.item()))

print("\nIf train loss keeps falling but val loss rises, the model is OVERFITTING.")
```

## 📤 Expected Output

```text
epoch | train loss | val loss
    1 |    45.0038 |   0.1379
   20 |     0.0492 |   0.0287
   40 |     0.1881 |   0.1248
   60 |     0.0554 |   0.0610
   80 |     0.0903 |   0.2418
  100 |     0.0478 |   0.0222

Best validation loss 0.0213 at epoch 38
Learned weight: 2.02 (true 2), bias: 0.93 (true 1)

If train loss keeps falling but val loss rises, the model is OVERFITTING.
```

## 🔍 Explanation
1. 50 samples are split into 40 train and 10 validation.
2. Each epoch has a training part and a validation part.
3. Validation uses `model.eval()` and `torch.no_grad()`.
4. The best validation loss and its epoch are remembered.
5. A table prints train and validation loss every 20 epochs.

---

## 📐 Typical Splits

| Set | Share | Purpose |
|-----|-------|---------|
| Train | ~70-80% | learn weights |
| Validation | ~10-20% | tune and monitor |
| Test | ~10-20% | final score |

## 📝 Important Points

- ✅ Never train on validation data.
- ✅ Falling training loss with rising validation loss means **overfitting**.
- ✅ Use `model.eval()` + `no_grad` when validating.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Training on validation data | Never let the model learn from it |
| Forgetting `model.eval()` | Switch modes before validating |

## 🧪 Try It Yourself

- [ ] Print the training loss as well.
- [ ] Change the split to 30/20.
- [ ] Train longer and watch the validation loss.

## ❓ Quick Questions

1. Why do we need validation data?
2. What does `model.eval()` do?
3. What is overfitting?

---

## 🏁 Summary

> 💡 Validation shows how the model behaves on data it has not learned from.

⬅️ **Previous:** [Training Loop](../03_training_loop/training_loop.md)  |  ➡️ **Next:** [Save Load Model](../05_save_load_model/save_load_model.md)
