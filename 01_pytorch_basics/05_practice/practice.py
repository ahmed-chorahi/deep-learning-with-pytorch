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
