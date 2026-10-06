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
