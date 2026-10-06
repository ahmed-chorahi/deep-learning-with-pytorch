# 🏋️ Dataloader

> 📚 **Training** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Learn how to batch and shuffle data.

## 📌 What is it?
A `DataLoader` takes a Dataset and serves it in **batches**, optionally in random order. Batches are faster to process and make training smoother.

## 🧠 Real-Life Analogy
Like dealing cards: the whole deck (dataset) is shuffled and then handed out a few cards (batch) at a time.

---

## 💻 Example

Code from `dataloader.py` (run it with `python dataloader.py`):

```python
# Lesson 2: DataLoader
# A DataLoader hands out the dataset in small groups called batches.

import torch
from torch.utils.data import Dataset, DataLoader


# Same dataset as in 01_dataset/dataset.py (copied so this lesson runs on its own)
class NumbersDataset(Dataset):
    def __init__(self, n=10):
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        self.y = 2 * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


torch.manual_seed(0)
dataset = NumbersDataset(n=10)

print("=== 1. Batches of 4 (no shuffle) ===")
loader = DataLoader(dataset, batch_size=4, shuffle=False)
print("number of batches:", len(loader))      # 10 / 4 -> 3 batches
for i, (x, y) in enumerate(loader):
    print(f"batch {i}: x = {x.flatten().tolist()}  (shape {tuple(x.shape)})")

print("\n=== 2. Shuffling (new order every epoch) ===")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for epoch in range(2):
    order = []
    for x, y in loader:
        order += x.flatten().int().tolist()
    print(f"epoch {epoch}: order = {order}")

print("\n=== 3. drop_last=True throws away the small last batch ===")
loader = DataLoader(dataset, batch_size=4, drop_last=True)
print("number of batches:", len(loader))

print("\n=== 4. Different batch sizes ===")
for size in [1, 2, 5, 10]:
    loader = DataLoader(dataset, batch_size=size)
    print(f"batch_size = {size:2d} -> {len(loader):2d} batches per epoch")

print("\n=== 5. Getting just one batch ===")
x, y = next(iter(DataLoader(dataset, batch_size=3)))
print("x:", x.flatten().tolist(), "| y:", y.flatten().tolist())

print("\n=== 6. A tiny summary after one pass ===")
loader = DataLoader(dataset, batch_size=4)
total = 0
for x, y in loader:
    total += len(x)
print("samples seen in one epoch:", total)
```

## 📤 Expected Output

```text
=== 1. Batches of 4 (no shuffle) ===
number of batches: 3
batch 0: x = [0.0, 1.0, 2.0, 3.0]  (shape (4, 1))
batch 1: x = [4.0, 5.0, 6.0, 7.0]  (shape (4, 1))
batch 2: x = [8.0, 9.0]  (shape (2, 1))

=== 2. Shuffling (new order every epoch) ===
epoch 0: order = [3, 5, 0, 6, 1, 2, 4, 9, 7, 8]
epoch 1: order = [2, 4, 9, 8, 7, 5, 6, 1, 0, 3]

=== 3. drop_last=True throws away the small last batch ===
number of batches: 2

=== 4. Different batch sizes ===
batch_size =  1 -> 10 batches per epoch
batch_size =  2 ->  5 batches per epoch
batch_size =  5 ->  2 batches per epoch
batch_size = 10 ->  1 batches per epoch

=== 5. Getting just one batch ===
x: [0.0, 1.0, 2.0] | y: [1.0, 3.0, 5.0]

=== 6. A tiny summary after one pass ===
samples seen in one epoch: 10
```

## 🔍 Explanation
1. Batches of 4 from 10 samples → 3 batches (4, 4, 2).
2. With `shuffle=True` the order changes every epoch.
3. `drop_last=True` removes the small last batch.
4. A loop compares different batch sizes.
5. `next(iter(loader))` grabs a single batch.

---

## 📐 Batch Shapes

```
dataset : 10 samples
batch   : x -> (4, 1), y -> (4, 1)
last    : x -> (2, 1)   (smaller, since 10 is not divisible by 4)
```

## 📝 Important Points

- ✅ Shuffle training data, not test data.
- ✅ `drop_last=True` removes the smaller last batch.
- ✅ Common batch sizes: 16, 32, 64.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Shuffling test data | Only shuffle training data |
| Too big a batch size | Try 16, 32 or 64 |

## 🧪 Try It Yourself

- [ ] Change `batch_size` to 5 and count the batches.
- [ ] Set `shuffle=False` and see the order.
- [ ] Use `drop_last=True`.

## ❓ Quick Questions

1. What does a DataLoader do?
2. Why shuffle?
3. How many batches for 10 samples and batch_size 4?

---

## 🏁 Summary

> 💡 A DataLoader turns a Dataset into shuffled batches.

⬅️ **Previous:** [Dataset](../01_dataset/dataset.md)  |  ➡️ **Next:** [Training Loop](../03_training_loop/training_loop.md)
