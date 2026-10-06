# 🏋️ Dataset

> 📚 **Training** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Learn how to wrap data in a PyTorch `Dataset`.

## 📌 What is it?
A `Dataset` tells PyTorch two things: how many samples there are (`__len__`) and how to get one sample (`__getitem__`).

## 🧠 Real-Life Analogy
Like a library catalogue: you can ask 'how many books?' and 'give me book number 3'.

---

## 💻 Example

Code from `dataset.py` (run it with `python dataset.py`):

```python
# Lesson 1: Dataset
# A Dataset tells PyTorch (1) how many samples we have and (2) how to get one sample.

import torch
from torch.utils.data import Dataset


class NumbersDataset(Dataset):
    """A tiny dataset where the answer follows y = 2x + 1 (plus optional noise)."""

    def __init__(self, n=10, noise=0.0):
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1)   # shape (n, 1)
        self.y = 2 * self.x + 1
        if noise > 0:
            self.y = self.y + noise * torch.randn(n, 1)              # make it a bit messy

    def __len__(self):
        return len(self.x)                    # how many samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]   # one (input, target) pair


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== 1. Clean dataset ===")
    dataset = NumbersDataset(n=5)
    print("size:", len(dataset))
    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"sample {i}: x = {x.item():.1f}, y = {y.item():.1f}")

    print("\n=== 2. Noisy dataset ===")
    noisy = NumbersDataset(n=5, noise=0.5)
    for i in range(len(noisy)):
        x, y = noisy[i]
        print(f"sample {i}: x = {x.item():.1f}, y = {y.item():.2f}")

    print("\n=== 3. Shapes ===")
    x, y = dataset[0]
    print("one x shape:", tuple(x.shape), "| one y shape:", tuple(y.shape))

    print("\n=== 4. Last sample using a negative index ===")
    print(dataset[-1])

    print("\n=== 5. Splitting into train and test ===")
    full = NumbersDataset(n=10)
    train_set, test_set = torch.utils.data.random_split(full, [8, 2])
    print("train size:", len(train_set), "| test size:", len(test_set))
```

## 📤 Expected Output

```text
=== 1. Clean dataset ===
size: 5
sample 0: x = 0.0, y = 1.0
sample 1: x = 1.0, y = 3.0
sample 2: x = 2.0, y = 5.0
sample 3: x = 3.0, y = 7.0
sample 4: x = 4.0, y = 9.0

=== 2. Noisy dataset ===
sample 0: x = 0.0, y = 1.77
sample 1: x = 1.0, y = 2.85
sample 2: x = 2.0, y = 3.91
sample 3: x = 3.0, y = 7.28
sample 4: x = 4.0, y = 8.46

=== 3. Shapes ===
one x shape: (1,) | one y shape: (1,)

=== 4. Last sample using a negative index ===
(tensor([4.]), tensor([9.]))

=== 5. Splitting into train and test ===
train size: 8 | test size: 2
```

## 🔍 Explanation
1. `NumbersDataset` stores `x` and `y = 2x + 1`, with optional noise.
2. `__len__` returns the size; `__getitem__` returns one pair.
3. Examples print clean and noisy samples and their shapes.
4. `random_split` divides the data into train and test sets.

---

## 📐 Shapes

```
x : (n, 1)
y : (n, 1)
dataset[i] -> (x[i], y[i])   each of shape (1,)
```

## 📝 Important Points

- ✅ Every custom Dataset needs `__len__` and `__getitem__`.
- ✅ A Dataset gives ONE sample; a DataLoader makes batches.
- ✅ Real datasets load files in `__getitem__`.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting `__len__` | DataLoader needs it |
| Returning lists instead of tensors | Return tensors |

## 🧪 Try It Yourself

- [ ] Change the formula to `y = 3x - 2`.
- [ ] Print `dataset[9]`.
- [ ] Add a method that returns the biggest x.

## ❓ Quick Questions

1. Which two methods does a Dataset need?
2. What does `__getitem__` return?
3. Does a Dataset create batches?

---

## 🏁 Summary

> 💡 A Dataset gives PyTorch one sample at a time.

➡️ **Next:** [Dataloader](../02_dataloader/dataloader.md)
