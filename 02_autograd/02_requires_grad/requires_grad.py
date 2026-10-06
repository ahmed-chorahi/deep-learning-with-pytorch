# Lesson 2: requires_grad
# A switch that tells PyTorch whether to record operations for gradients.

import torch
import torch.nn as nn

print("=== 1. Turning tracking on ===")
a = torch.tensor([1.0, 2.0])
print("a.requires_grad:", a.requires_grad)          # False by default

b = torch.tensor([1.0, 2.0], requires_grad=True)
print("b.requires_grad:", b.requires_grad)

a.requires_grad_(True)                              # turn on later
print("a.requires_grad now:", a.requires_grad)

print("\n=== 2. Results of tracked tensors are tracked ===")
c = b * 2
print("c.requires_grad:", c.requires_grad)
print("c.grad_fn:", c.grad_fn)

print("\n=== 3. torch.no_grad() for predictions ===")
with torch.no_grad():
    d = b * 2
    print("d.requires_grad inside no_grad:", d.requires_grad)

print("\n=== 4. detach() ===")
e = (b * 2).detach()
print("e.requires_grad:", e.requires_grad)
print("e as numpy:", e.numpy())                     # only works without tracking

print("\n=== 5. Model weights have requires_grad=True ===")
layer = nn.Linear(3, 1)
for name, p in layer.named_parameters():
    print(f"{name:7s} requires_grad = {p.requires_grad}")

print("\n=== 6. Freezing a layer (turn learning off) ===")
for p in layer.parameters():
    p.requires_grad = False
x = torch.rand(2, 3)
out = layer(x).sum()
print("output requires_grad:", out.requires_grad)

print("\n=== 7. Trying to use grad on a non-float tensor ===")
try:
    torch.tensor([1, 2, 3], requires_grad=True)
except RuntimeError as error:
    print("Error:", str(error)[:70])
