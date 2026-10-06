# 🔢 Tensor Shapes

> 📚 **PyTorch Basics** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Learn to read and change the shape of a tensor.

## 📌 What is it?
The **shape** of a tensor describes its size in every dimension. Most PyTorch errors are shape errors, so changing shapes with `reshape`, `unsqueeze`, `squeeze`, `permute` and `flatten` is a key skill.

## 🧠 Real-Life Analogy
Think of 12 eggs. You can lay them in a 3x4 grid, a 2x6 grid, or one long row. The eggs (data) never change, only the arrangement (shape).

---

## 💻 Example

Code from `tensor_shapes.py` (run it with `python tensor_shapes.py`):

```python
# Lesson 2: Tensor shapes
# Most PyTorch errors are shape errors, so learn to change shapes confidently.

import torch


def show(name, t):
    # small helper so every print looks the same
    print(f"{name:22s} shape = {tuple(t.shape)}")


x = torch.arange(12)
show("original", x)

print("\n=== reshape ===")
show("reshape(3, 4)", x.reshape(3, 4))
show("reshape(2, 6)", x.reshape(2, 6))
show("reshape(2, -1)", x.reshape(2, -1))     # -1 = "you work it out"
show("reshape(2, 3, 2)", x.reshape(2, 3, 2))

# Wrong reshape: 12 numbers cannot fit into 5 x 3 = 15 places
try:
    x.reshape(5, 3)
except RuntimeError as error:
    print("Error caught:", str(error)[:60], "...")

print("\n=== squeeze and unsqueeze ===")
v = torch.tensor([1, 2, 3])
show("v", v)
show("v.unsqueeze(0)", v.unsqueeze(0))       # (1, 3)  one row
show("v.unsqueeze(1)", v.unsqueeze(1))       # (3, 1)  one column
show("zeros(1,3,1).squeeze()", torch.zeros(1, 3, 1).squeeze())

print("\n=== transpose and permute ===")
m = torch.rand(2, 3)
show("m", m)
show("m.T", m.T)
image = torch.rand(28, 28, 3)                # height, width, channels
show("image (H, W, C)", image)
show("permute(2, 0, 1)", image.permute(2, 0, 1))   # PyTorch wants (C, H, W)

print("\n=== flatten ===")
show("flatten of (2,3,4)", torch.rand(2, 3, 4).flatten())

print("\n=== batches ===")
one_image = torch.rand(1, 28, 28)
batch = one_image.unsqueeze(0)               # models expect a batch dimension
show("one image", one_image)
show("batch of 1 image", batch)
show("batch of 64 images", torch.rand(64, 1, 28, 28))
```

## 📤 Expected Output

```text
original               shape = (12,)

=== reshape ===
reshape(3, 4)          shape = (3, 4)
reshape(2, 6)          shape = (2, 6)
reshape(2, -1)         shape = (2, 6)
reshape(2, 3, 2)       shape = (2, 3, 2)
Error caught: shape '[5, 3]' is invalid for input of size 12 ...

=== squeeze and unsqueeze ===
v                      shape = (3,)
v.unsqueeze(0)         shape = (1, 3)
v.unsqueeze(1)         shape = (3, 1)
zeros(1,3,1).squeeze() shape = (3,)

=== transpose and permute ===
m                      shape = (2, 3)
m.T                    shape = (3, 2)
image (H, W, C)        shape = (28, 28, 3)
permute(2, 0, 1)       shape = (3, 28, 28)

=== flatten ===
flatten of (2,3,4)     shape = (24,)

=== batches ===
one image              shape = (1, 28, 28)
batch of 1 image       shape = (1, 1, 28, 28)
batch of 64 images     shape = (64, 1, 28, 28)

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. A small `show()` helper prints names and shapes in a neat column.
2. `reshape` changes 12 numbers into several grids; `-1` is worked out automatically.
3. A wrong reshape (5×3 for 12 numbers) is caught with `try/except`.
4. `unsqueeze` and `squeeze` add and remove size-1 dimensions.
5. `.T` and `permute` reorder dimensions, e.g. image `(H, W, C)` → `(C, H, W)`.
6. `flatten` makes one long vector.
7. The last part builds batch shapes like `(64, 1, 28, 28)`.

---

## 📐 Common Shapes in Deep Learning

| Data | Typical shape |
|------|---------------|
| One grayscale image | `(1, 28, 28)` = channels, height, width |
| A batch of 64 images | `(64, 1, 28, 28)` |
| A batch of 32 samples with 10 features | `(32, 10)` |

Rule: the **total number of elements must stay the same** when you reshape.

## 📝 Important Points

- ✅ Reshape never changes the data, only how it is arranged.
- ✅ Use `-1` for one dimension that should be calculated automatically.
- ✅ `unsqueeze(0)` is the usual way to add a batch dimension.
- ✅ Always print `x.shape` when something looks wrong.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Reshaping to a size with a different number of elements | Check that the product of the sizes matches |
| Forgetting the batch dimension for one image | Use `unsqueeze(0)` |
| Feeding `(H, W, C)` images to PyTorch | Use `permute(2, 0, 1)` |

## 🧪 Try It Yourself

- [ ] Reshape `torch.arange(24)` into `(2, 3, 4)`.
- [ ] Use `unsqueeze` to turn shape `(5,)` into `(5, 1)`.
- [ ] What shape does `torch.rand(4, 5).T` have?

## ❓ Quick Questions

1. Why must the number of elements stay the same in `reshape`?
2. What does `squeeze()` do?
3. What shape do you get from `flatten()` on a `(2, 3, 4)` tensor?

---

## 🏁 Summary

> 💡 Always know your shape; reshape, squeeze, unsqueeze and permute fix most errors.

⬅️ **Previous:** [Tensors](../01_tensors/tensors.md)  |  ➡️ **Next:** [Tensor Operations](../03_tensor_operations/tensor_operations.md)
