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
