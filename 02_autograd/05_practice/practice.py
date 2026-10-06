# Practice for Section 2: autograd

import torch


def check(task, ok):
    print(f"Task {task}:", "PASS" if ok else "FAIL")


# Task 1: gradient of y = 5x^3 at x = 2  (dy/dx = 15x^2 = 60)
x = torch.tensor(2.0, requires_grad=True)
y = 5 * x ** 3
y.backward()
print("Task 1:", x.grad.item())
check(1, x.grad.item() == 60.0)

# Task 2: gradients of f = a*b + c
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = torch.tensor(1.0, requires_grad=True)
f = a * b + c
f.backward()
print("Task 2:", a.grad.item(), b.grad.item(), c.grad.item())
check(2, (a.grad.item(), b.grad.item(), c.grad.item()) == (4.0, 3.0, 1.0))

# Task 3: gradient descent on (w - 3)^2 + 2
w = torch.tensor(10.0, requires_grad=True)
for _ in range(50):
    loss = (w - 3) ** 2 + 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_()
print("Task 3: w =", round(w.item(), 3))
check(3, abs(w.item() - 3) < 0.01)

# Task 4: no_grad stops tracking
with torch.no_grad():
    r = x * 2
print("Task 4:", r.requires_grad)
check(4, r.requires_grad is False)

# Task 5: gradient of x^2 + y^2 at (1, 2)  -> (2, 4)
p = torch.tensor(1.0, requires_grad=True)
q = torch.tensor(2.0, requires_grad=True)
((p ** 2) + (q ** 2)).backward()
print("Task 5:", p.grad.item(), q.grad.item())
check(5, (p.grad.item(), q.grad.item()) == (2.0, 4.0))

# Task 6: compare two learning rates on (w - 5)^2
for lr in [0.01, 0.5]:
    w = torch.tensor(0.0, requires_grad=True)
    for _ in range(10):
        loss = (w - 5) ** 2
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
    print(f"Task 6: lr = {lr:<4} -> w after 10 steps = {w.item():.3f}")
print("Task 6: a small lr is slow, a big lr jumps closer (but too big can overshoot)")
