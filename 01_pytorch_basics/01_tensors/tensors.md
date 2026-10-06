# 🔢 PyTorch Tensors

> 📚 **PyTorch Basics** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Understand what tensors are and how they are used in PyTorch.

## 📌 What is a Tensor?
A **tensor** is PyTorch's basic container for numbers. It looks like a NumPy array, but it can also run on a GPU and keep track of gradients (which neural networks need to learn).

## 🧠 Real-Life Analogy
A tensor is like a box of numbers. A single number is a box with one item, a list is a row of boxes, a table is a grid of boxes, and a stack of tables is a 3D block of boxes.

---

## 💻 Example

Code from `tensors.py` (run it with `python tensors.py`):

```python
# Lesson 1: Tensors
# A tensor is PyTorch's version of a NumPy array.
# It stores numbers in a grid, and it can also run on a GPU and track gradients.

import torch
import numpy as np

print("=== 1. Tensors with different dimensions ===")
scalar = torch.tensor(7)                 # 0D: one number
vector = torch.tensor([1, 2, 3])         # 1D: a list of numbers
matrix = torch.tensor([[1, 2], [3, 4]])  # 2D: a table
cube = torch.zeros(2, 3, 4)              # 3D: a stack of tables

for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("cube", cube)]:
    print(f"{name:7s} dims = {t.dim()}  shape = {tuple(t.shape)}")

print("\n=== 2. Handy ways to create tensors ===")
print("zeros:\n", torch.zeros(2, 3))
print("ones:\n", torch.ones(2, 3))
print("range:", torch.arange(0, 10, 2))
print("evenly spaced:", torch.linspace(0, 1, 5))
print("filled with 9:\n", torch.full((2, 2), 9))

print("\n=== 3. Random tensors (use a seed to repeat results) ===")
torch.manual_seed(42)
print("random 0-1:", torch.rand(3))
print("random normal:", torch.randn(3))

print("\n=== 4. Data types ===")
a = torch.tensor([1, 2, 3])
b = torch.tensor([1.5, 2.5, 3.5])
print("a dtype:", a.dtype)             # int64
print("b dtype:", b.dtype)             # float32
print("a as float:", a.float())
print("b as int  :", b.int())          # cuts the decimals

print("\n=== 5. NumPy <-> PyTorch ===")
arr = np.array([10, 20, 30])
t = torch.from_numpy(arr)
print("NumPy to tensor:", t)
print("Tensor to NumPy:", t.numpy())

print("\n=== 6. Copying ===")
original = torch.tensor([1, 2, 3])
copy = original.clone()                # a real, separate copy
copy[0] = 99
print("original:", original, "| copy:", copy)

print("\n=== 7. CPU or GPU? ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
x = torch.ones(2).to(device)           # move a tensor to the device
print("x lives on:", x.device)
```

## 📤 Expected Output

```text
=== 1. Tensors with different dimensions ===
scalar  dims = 0  shape = ()
vector  dims = 1  shape = (3,)
matrix  dims = 2  shape = (2, 2)
cube    dims = 3  shape = (2, 3, 4)

=== 2. Handy ways to create tensors ===
zeros:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
ones:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])
range: tensor([0, 2, 4, 6, 8])
evenly spaced: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
filled with 9:
 tensor([[9, 9],
        [9, 9]])

=== 3. Random tensors (use a seed to repeat results) ===
random 0-1: tensor([0.8823, 0.9150, 0.3829])
random normal: tensor([-0.8293, -1.6137, -0.2147])

=== 4. Data types ===
a dtype: torch.int64
b dtype: torch.float32
a as float: tensor([1., 2., 3.])
b as int  : tensor([1, 2, 3], dtype=torch.int32)

=== 5. NumPy <-> PyTorch ===
NumPy to tensor: tensor([10, 20, 30])
Tensor to NumPy: [10 20 30]

=== 6. Copying ===
original: tensor([1, 2, 3]) | copy: tensor([99,  2,  3])

=== 7. CPU or GPU? ===
Using device: cpu
x lives on: cpu
```

## 🔍 Explanation
1. Section 1 creates a scalar, vector, matrix and cube and prints their `dim()` and `shape`.
2. Section 2 uses `zeros`, `ones`, `arange`, `linspace` and `full` to build tensors quickly.
3. Section 3 makes random tensors; `manual_seed(42)` makes them repeatable.
4. Section 4 shows data types (`int64` vs `float32`) and converting with `.float()` / `.int()`.
5. Section 5 converts between NumPy and PyTorch.
6. Section 6 uses `clone()` to make a real copy.
7. Section 7 checks for a GPU and moves a tensor with `.to(device)`.

---

## 📐 Tensor Shape

The **shape** tells you how many numbers there are in each direction.

| Kind | Example | Shape |
|------|---------|-------|
| 0D tensor (scalar) | `torch.tensor(5)` | `torch.Size([])` |
| 1D tensor (vector) | `torch.tensor([1, 2, 3])` | `torch.Size([3])` |
| 2D tensor (matrix) | `torch.zeros(2, 3)` | `torch.Size([2, 3])` |
| 3D tensor | `torch.zeros(2, 3, 4)` | `torch.Size([2, 3, 4])` |

Check any tensor with `x.shape` (or `x.dim()` for the number of dimensions).

## 📝 Important Points

- ✅ Tensors hold numbers of one data type (float32 by default for decimals).
- ✅ Create them from lists, or with `zeros`, `ones`, `rand`, `arange`.
- ✅ Tensors can move to a GPU; Python lists cannot.
- ✅ `torch.from_numpy` shares memory with the NumPy array.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Mixing int and float tensors in maths | Convert with `.float()` first |
| Thinking `=` makes a copy | Use `.clone()` for a real copy |
| Forgetting the GPU may not exist | Choose the device with `torch.cuda.is_available()` |

## 🧪 Try It Yourself

- [ ] Create a 4x4 tensor of ones.
- [ ] Create a tensor with the even numbers from 0 to 20.
- [ ] Convert `torch.tensor([1.9, 2.7])` to integers. What do you get?

## ❓ Quick Questions

1. What is a tensor?
2. What is its shape?
3. How is it different from a Python list?

---

## 🏁 Summary

> 💡 A tensor is a grid of numbers with a shape, a data type and a device.

➡️ **Next:** [Tensor Shapes](../02_tensor_shapes/tensor_shapes.md)
