# 🔢 Indexing Slicing

> 📚 **PyTorch Basics** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Pick out single values, rows, columns and groups of values from a tensor.

## 📌 What is it?
**Indexing** picks one element and **slicing** picks a range. It works exactly like Python lists and NumPy, with an extra index for each dimension.

## 🧠 Real-Life Analogy
A tensor is like a chocolate bar. Indexing picks one square, slicing breaks off a row or a corner, and a boolean mask picks every square that has nuts.

---

## 💻 Example

Code from `indexing_slicing.py` (run it with `python indexing_slicing.py`):

```python
# Lesson 4: Indexing and slicing
# Each ROW is a student, each COLUMN is a subject (Maths, Physics, English).

import torch

marks = torch.tensor([[80, 70, 65],
                      [55, 90, 72],
                      [40, 45, 60],
                      [95, 85, 88]])
print("marks table (4 students x 3 subjects):\n", marks)

print("\n=== 1. Indexing ===")
print("student 0, all subjects:", marks[0])
print("student 1, physics     :", marks[1, 1])
print("last student           :", marks[-1])

print("\n=== 2. Slicing (start:stop, stop NOT included) ===")
print("first 2 students:\n", marks[:2])
print("maths column    :", marks[:, 0])
print("maths + physics:\n", marks[:, :2])
print("students 1 and 2, English:", marks[1:3, 2])

print("\n=== 3. Boolean masks ===")
mask = marks > 70
print("marks > 70:\n", mask)
print("those values:", marks[mask])
print("count:", mask.sum().item())

print("\n=== 4. Using conditions to find students ===")
maths = marks[:, 0]
print("maths marks:", maths)
print("students with maths > 60 (row numbers):", torch.where(maths > 60)[0])
print("best maths student is row", maths.argmax().item())

print("\n=== 5. Changing values ===")
curved = marks.clone()
curved[2, 0] = 50                       # fix one mark
curved[curved < 50] = 50                # minimum mark is 50
print("after changes:\n", curved)

print("\n=== 6. Index with a list ===")
print("students 0 and 3:\n", marks[[0, 3]])

print("\n=== 7. Averages ===")
print("average per student:", marks.float().mean(dim=1))
print("average per subject:", marks.float().mean(dim=0))
```

## 📤 Expected Output

```text
marks table (4 students x 3 subjects):
 tensor([[80, 70, 65],
        [55, 90, 72],
        [40, 45, 60],
        [95, 85, 88]])

=== 1. Indexing ===
student 0, all subjects: tensor([80, 70, 65])
student 1, physics     : tensor(90)
last student           : tensor([95, 85, 88])

=== 2. Slicing (start:stop, stop NOT included) ===
first 2 students:
 tensor([[80, 70, 65],
        [55, 90, 72]])
maths column    : tensor([80, 55, 40, 95])
maths + physics:
 tensor([[80, 70],
        [55, 90],
        [40, 45],
        [95, 85]])
students 1 and 2, English: tensor([72, 60])

=== 3. Boolean masks ===
marks > 70:
 tensor([[ True, False, False],
        [False,  True,  True],
        [False, False, False],
        [ True,  True,  True]])
those values: tensor([80, 90, 72, 95, 85, 88])
count: 6

=== 4. Using conditions to find students ===
maths marks: tensor([80, 55, 40, 95])
students with maths > 60 (row numbers): tensor([0, 3])
best maths student is row 3

=== 5. Changing values ===
after changes:
 tensor([[80, 70, 65],
        [55, 90, 72],
        [50, 50, 60],
        [95, 85, 88]])

=== 6. Index with a list ===
students 0 and 3:
 tensor([[80, 70, 65],
        [95, 85, 88]])

=== 7. Averages ===
average per student: tensor([71.6667, 72.3333, 48.3333, 89.3333])
average per subject: tensor([67.5000, 72.5000, 71.2500])
```

## 🔍 Explanation
1. A 4×3 `marks` table: rows are students, columns are subjects.
2. Indexing picks one row or value; negative index counts from the end.
3. Slicing (`:2`, `:, 0`) picks ranges; the stop is not included.
4. A boolean mask (`marks > 70`) filters values.
5. `torch.where` and `argmax` find which student matches a condition.
6. Assigning to a slice or mask changes values; `clone()` protects the original.
7. `mean(dim=...)` gives averages per student and per subject.

---

## 📐 Index Cheat Sheet

| Expression | Meaning |
|-----------|---------|
| `x[0]` | first row |
| `x[-1]` | last row |
| `x[:, 1]` | second column |
| `x[1:3]` | rows 1 and 2 |
| `x[:, ::2]` | every other column |
| `x[x > 5]` | all values greater than 5 |

## 📝 Important Points

- ✅ Counting starts at 0; `-1` is the last item.
- ✅ Slices do not include the stop index.
- ✅ Masks are a powerful way to filter data.
- ✅ Basic slices share memory with the original tensor.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Expecting the stop index to be included | `x[1:3]` gives rows 1 and 2 only |
| Changing a slice and being surprised the original changed | Use `.clone()` first |
| Mixing up rows and columns | `x[row, column]` |

## 🧪 Try It Yourself

- [ ] Get the bottom-right 2x2 block of a 3x3 tensor.
- [ ] Set every value below 3 to zero using a mask.
- [ ] Pick rows 0 and 2 with a list.

## ❓ Quick Questions

1. What does `x[:, 0]` return?
2. What does `x[1:3]` include?
3. How do you select all values greater than 5?

---

## 🏁 Summary

> 💡 Indexing, slicing and masks let you pick exactly the data you need.

⬅️ **Previous:** [Tensor Operations](../03_tensor_operations/tensor_operations.md)  |  ➡️ **Next:** [Practice](../05_practice/practice.md)
