# 🔢 Practice

> 📚 **PyTorch Basics** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Practice everything from Section 1: creating, reshaping, calculating and indexing tensors.

## 📌 What is it?
This lesson is a set of 7 small tasks. Try each one on your own first, then compare with the answer in the code.

## 🧠 Real-Life Analogy
It's like the exercises at the end of a textbook chapter. Reading is not enough; you learn by doing.

---

## 💻 Example

Code from `practice.py` (run it with `python practice.py`):

```python
# Practice for Section 1
# Try each task yourself first. The code shows one possible answer.

import torch


def check(task, ok):
    print(f"Task {task}:", "PASS" if ok else "FAIL")


# Task 1: numbers 1 to 6 reshaped to (2, 3)
t = torch.arange(1, 7).reshape(2, 3)
print("Task 1:\n", t)
check(1, t.shape == (2, 3))

# Task 2: multiply every element by 2 and add 1
r = t * 2 + 1
print("Task 2:\n", r)
check(2, r[0, 0].item() == 3)

# Task 3: sum of each column
col_sums = t.sum(dim=0)
print("Task 3:", col_sums)
check(3, col_sums.tolist() == [5, 7, 9])

# Task 4: second row, last element
value = t[1, -1]
print("Task 4:", value.item())
check(4, value.item() == 6)

# Task 5: 3x3 ones with 5 in the middle
m = torch.ones(3, 3)
m[1, 1] = 5
print("Task 5:\n", m)
check(5, m.sum().item() == 13)

# Task 6: matrix multiplication shape
a = torch.rand(2, 3)
b = torch.rand(3, 2)
print("Task 6 shape:", (a @ b).shape)
check(6, (a @ b).shape == (2, 2))

# Task 7: values greater than 3
big = t[t > 3]
print("Task 7:", big)
check(7, big.tolist() == [4, 5, 6])

# Task 8: average of the diagonal of a 3x3 matrix
d = torch.arange(9).reshape(3, 3).float()
diag_mean = torch.diag(d).mean()
print("Task 8:", diag_mean.item())
check(8, diag_mean.item() == 4.0)

# Task 9: normalize [10, 20, 30, 40] to the range 0..1
x = torch.tensor([10.0, 20.0, 30.0, 40.0])
scaled = (x - x.min()) / (x.max() - x.min())
print("Task 9:", scaled)
check(9, scaled.tolist() == [0.0, 1 / 3, 2 / 3, 1.0] or abs(scaled[1].item() - 0.3333) < 0.001)
```

## 📤 Expected Output

```text
Task 1:
 tensor([[1, 2, 3],
        [4, 5, 6]])
Task 1: PASS
Task 2:
 tensor([[ 3,  5,  7],
        [ 9, 11, 13]])
Task 2: PASS
Task 3: tensor([5, 7, 9])
Task 3: PASS
Task 4: 6
Task 4: PASS
Task 5:
 tensor([[1., 1., 1.],
        [1., 5., 1.],
        [1., 1., 1.]])
Task 5: PASS
Task 6 shape: torch.Size([2, 2])
Task 6: PASS
Task 7: tensor([4, 5, 6])
Task 7: PASS
Task 8: 4.0
Task 8: PASS
Task 9: tensor([0.0000, 0.3333, 0.6667, 1.0000])
Task 9: PASS

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. A `check()` helper prints PASS or FAIL for every task.
2. Tasks 1-3: reshape, arithmetic and column sums.
3. Tasks 4-5: indexing and changing one value.
4. Tasks 6-7: matrix multiplication shape and a boolean mask.
5. Task 8: average of a matrix diagonal with `torch.diag`.
6. Task 9: scale numbers to the range 0..1 (min-max scaling).

---

## 📐 Shapes Used Here

```
t       -> (2, 3)
a @ b   -> (2, 3) @ (3, 2) = (2, 2)
diag    -> (3,)
```

## 📝 Important Points

- ✅ Cover the code and try each task before running it.
- ✅ Print `.shape` often.
- ✅ If a task fails, read the error message; it usually names the problem.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Looking at the answer too early | Try for 5 minutes first |
| Not printing shapes | Print `.shape` when a result looks odd |

## 🧪 Try It Yourself

- [ ] Write 3 new tasks of your own and solve them.
- [ ] Change Task 5 to put a 9 in every corner.
- [ ] Reshape `t` to `(3, 2)` and redo Task 3.

## ❓ Quick Questions

1. How do you reshape `arange(1, 7)` to `(2, 3)`?
2. What shape comes out of `(2, 3) @ (3, 2)`?
3. How do you select values greater than 3?

---

## 🏁 Summary

> 💡 Nine small tasks that check you can create, reshape, calculate and index tensors.

⬅️ **Previous:** [Indexing Slicing](../04_indexing_slicing/indexing_slicing.md)
