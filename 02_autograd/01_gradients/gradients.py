# Lesson 1: Gradients with autograd
# A gradient tells us how much the output changes when the input changes a little.
# PyTorch's autograd calculates it for us, so we don't do calculus by hand.

import torch

print("=== 1. y = x^2   (derivative: 2x) ===")
x = torch.tensor(3.0, requires_grad=True)   # "please track this tensor"
y = x ** 2
y.backward()                                # calculate dy/dx
print("x =", x.item())
print("dy/dx =", x.grad.item(), "(we expect 2 * 3 = 6)")

print("\n=== 2. z = 3x^2 + 2x + 1   (derivative: 6x + 2) ===")
x = torch.tensor(2.0, requires_grad=True)
z = 3 * x ** 2 + 2 * x + 1
z.backward()
print("dz/dx at x=2 is", x.grad.item(), "(we expect 6*2 + 2 = 14)")

print("\n=== 3. Two variables: f = a*b + b^2 ===")
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
f = a * b + b ** 2
f.backward()
print("df/da =", a.grad.item(), "(expected b = 5)")
print("df/db =", b.grad.item(), "(expected a + 2b = 12)")

print("\n=== 4. Gradients of a vector (use sum to get one number) ===")
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (w ** 2).sum()
loss.backward()
print("w    :", w.data)
print("grad :", w.grad, "(expected 2 * w)")

print("\n=== 5. Checking autograd against a tiny numeric estimate ===")
def f_of_x(value):
    return 3 * value ** 2 + 2 * value + 1

h = 0.0001
numeric = (f_of_x(2.0 + h) - f_of_x(2.0 - h)) / (2 * h)
print("numeric estimate:", round(numeric, 3))
print("autograd answer :", 14.0)
