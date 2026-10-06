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
