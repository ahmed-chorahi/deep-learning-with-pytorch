# 🔢 Tensor Operations

> 📚 **PyTorch Basics** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
Do math with tensors: arithmetic, reductions and matrix multiplication.

## 📌 What is it?
PyTorch can add, multiply and transform whole tensors at once, without writing loops. Operations are either **element-wise** (number by number) or **reductions** (many numbers become one).

## 🧠 Real-Life Analogy
Imagine two rows of numbers on a spreadsheet. Adding the rows adds each pair of cells. Taking a total at the bottom is a reduction.

---

## 💻 Example

Code from `tensor_operations.py` (run it with `python tensor_operations.py`):

```python
# Lesson 3: Tensor operations
# Math on whole tensors without writing loops.

import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("=== 1. Element-wise math ===")
print("a + b  =", a + b)
print("a * b  =", a * b)
print("a ** 2 =", a ** 2)
print("a / b  =", a / b)

print("\n=== 2. Broadcasting (different shapes working together) ===")
print("a + 10 =", a + 10)
table = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
print("table + a:\n", table + a)             # a is added to every row

print("\n=== 3. Reductions ===")
print("sum :", a.sum().item())
print("mean:", a.mean().item())
print("max :", a.max().item(), "at index", a.argmax().item())
print("column sums (dim=0):", table.sum(dim=0))
print("row sums    (dim=1):", table.sum(dim=1))
print("row means   (dim=1):", table.mean(dim=1))

print("\n=== 4. Matrix multiplication ===")
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print("A @ B:\n", A @ B)
print("A * B (not the same!):\n", A * B)
print("dot product of a and b:", torch.dot(a, b).item())

print("\n=== 5. Comparisons ===")
marks = torch.tensor([45, 78, 62, 91, 33])
print("marks:", marks)
print("passed (>= 50):", marks >= 50)
print("how many passed:", (marks >= 50).sum().item())

print("\n=== 6. Useful functions ===")
x = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
print("abs  :", x.abs())
print("clamp (limit to 0..2):", x.clamp(min=0, max=2))
print("sqrt of a:", a.sqrt())

print("\n=== 7. Joining tensors ===")
print("cat  :", torch.cat([a, b]))
print("stack:\n", torch.stack([a, b]))

print("\n=== 8. Mini example: normalize marks ===")
marks = marks.float()
normalized = (marks - marks.mean()) / marks.std()
print("normalized:", normalized)
print("new mean (~0):", round(normalized.mean().item(), 4))
```

## 📤 Expected Output

```text
=== 1. Element-wise math ===
a + b  = tensor([5., 7., 9.])
a * b  = tensor([ 4., 10., 18.])
a ** 2 = tensor([1., 4., 9.])
a / b  = tensor([0.2500, 0.4000, 0.5000])

=== 2. Broadcasting (different shapes working together) ===
a + 10 = tensor([11., 12., 13.])
table + a:
 tensor([[2., 4., 6.],
        [5., 7., 9.]])

=== 3. Reductions ===
sum : 6.0
mean: 2.0
max : 3.0 at index 2
column sums (dim=0): tensor([5., 7., 9.])
row sums    (dim=1): tensor([ 6., 15.])
row means   (dim=1): tensor([2., 5.])

=== 4. Matrix multiplication ===
A @ B:
 tensor([[19., 22.],
        [43., 50.]])
A * B (not the same!):
 tensor([[ 5., 12.],
        [21., 32.]])
dot product of a and b: 32.0

=== 5. Comparisons ===
marks: tensor([45, 78, 62, 91, 33])
passed (>= 50): tensor([False,  True,  True,  True, False])
how many passed: 3

=== 6. Useful functions ===
abs  : tensor([2.0000, 0.5000, 0.0000, 1.5000, 3.0000])
clamp (limit to 0..2): tensor([0.0000, 0.0000, 0.0000, 1.5000, 2.0000])
sqrt of a: tensor([1.0000, 1.4142, 1.7321])

=== 7. Joining tensors ===
cat  : tensor([1., 2., 3., 4., 5., 6.])
stack:
 tensor([[1., 2., 3.],
        [4., 5., 6.]])

=== 8. Mini example: normalize marks ===
normalized: tensor([-0.7120,  0.6866,  0.0085,  1.2376, -1.2206])
new mean (~0): 0.0
```

## 🔍 Explanation
1. Element-wise maths: `+`, `*`, `**`, `/` work on every pair of numbers.
2. Broadcasting adds a single number or a row to a whole tensor.
3. Reductions (`sum`, `mean`, `max`, `argmax`) with `dim=0` (columns) and `dim=1` (rows).
4. `@` is matrix multiplication; `*` is not.
5. Comparisons like `marks >= 50` give True/False tensors you can count.
6. `abs`, `clamp` and `sqrt` are handy functions.
7. `cat` and `stack` join tensors; the last part normalizes marks using mean and standard deviation.

---

## 📐 Shapes in Matrix Multiplication

For `A @ B` the inner sizes must match:

```
(2, 3) @ (3, 4) -> (2, 4)
```

`dim=0` collapses the **rows** (you get one number per column), and `dim=1` collapses the **columns** (one number per row).

## 📝 Important Points

- ✅ `*` multiplies element by element; `@` is true matrix multiplication.
- ✅ Reductions can be done along a chosen `dim`.
- ✅ `argmax` gives the *position* of the maximum, not the value.
- ✅ `cat` joins along an existing dimension, `stack` creates a new one.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Using `*` when you want matrix multiplication | Use `@` |
| Mixing up `dim=0` and `dim=1` | dim=0 → down the rows (per column), dim=1 → across (per row) |
| Integer tensors in `mean()` | Convert with `.float()` first |

## 🧪 Try It Yourself

- [ ] Compute the mean of `[2., 4., 6., 8.]`.
- [ ] Multiply a `(2, 3)` tensor by a `(3, 2)` tensor.
- [ ] What is the difference between `cat` and `stack`?

## ❓ Quick Questions

1. What is the difference between `*` and `@`?
2. What does `sum(dim=0)` do on a 2D tensor?
3. What is broadcasting?

---

## 🏁 Summary

> 💡 Tensor maths avoids loops: element-wise, broadcasting, reductions and matrix multiplication.

⬅️ **Previous:** [Tensor Shapes](../02_tensor_shapes/tensor_shapes.md)  |  ➡️ **Next:** [Indexing Slicing](../04_indexing_slicing/indexing_slicing.md)
