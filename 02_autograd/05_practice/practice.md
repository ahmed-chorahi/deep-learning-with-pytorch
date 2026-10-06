# 📈 Practice

> 📚 **Autograd** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Practice computing gradients and running gradient descent.

## 📌 What is it?
Four short tasks that combine everything from autograd: simple gradients, several variables, a gradient descent loop and `no_grad`.

## 🧠 Real-Life Analogy
Like practicing scales on a piano: small repeated exercises make the main skills automatic.

---

## 💻 Example

Code from `practice.py` (run it with `python practice.py`):

```python
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
```

## 📤 Expected Output

```text
Task 1: 60.0
Task 1: PASS
Task 2: 4.0 3.0 1.0
Task 2: PASS
Task 3: w = 3.0
Task 3: PASS
Task 4: False
Task 4: PASS
Task 5: 2.0 4.0
Task 5: PASS
Task 6: lr = 0.01 -> w after 10 steps = 0.915
Task 6: lr = 0.5  -> w after 10 steps = 5.000
Task 6: a small lr is slow, a big lr jumps closer (but too big can overshoot)
```

## 🔍 Explanation
1. Task 1-2: gradients of `5x³` and `a*b + c`.
2. Task 3: gradient descent finds the minimum of `(w-3)² + 2`.
3. Task 4: `no_grad` stops tracking.
4. Task 5: gradients of `x² + y²` at (1, 2).
5. Task 6: compares learning rates 0.01 and 0.5.

---

## 📐 Answers to Check

| Task | Expected |
|------|----------|
| 1 | 60.0 |
| 2 | 4.0, 3.0, 1.0 |
| 3 | w close to 3 |
| 4 | False |
| 5 | 2.0, 4.0 |
| 6 | lr 0.01 is slow, lr 0.5 reaches 5 |

## 📝 Important Points

- ✅ Work out the derivative on paper first, then check with PyTorch.
- ✅ Always reset gradients inside loops.
- ✅ Gradient descent works for any function you can compute.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Using a learning rate that is too big | Try smaller values if the loss jumps |
| Forgetting to zero gradients in loops | Reset after each update |

## 🧪 Try It Yourself

- [ ] Find the minimum of `(w - 8)² + 1` by gradient descent.
- [ ] Compute the gradient of `x² + y²` at (1, 2).
- [ ] Try different learning rates in Task 3.

## ❓ Quick Questions

1. What is d/dx of 5x³?
2. What does Task 3 minimize?
3. What does `no_grad` do?

---

## 🏁 Summary

> 💡 Six tasks that practice gradients, gradient descent and learning rates.

⬅️ **Previous:** [Computational Graph](../04_computational_graph/computational_graph.md)
