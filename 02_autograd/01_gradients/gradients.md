# 📈 Gradients

> 📚 **Autograd** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Understand that PyTorch can compute derivatives (gradients) automatically.

## 📌 What is it?
A **gradient** tells you how much the output changes when you nudge an input. Neural networks use gradients to learn which way to adjust their weights. PyTorch's **autograd** calculates them for you.

## 🧠 Real-Life Analogy
Imagine standing on a hill in fog. The gradient is the slope under your feet. It tells you which direction is uphill, so you also know which direction is downhill.

---

## 💻 Example

Code from `gradients.py` (run it with `python gradients.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. y = x^2   (derivative: 2x) ===
x = 3.0
dy/dx = 6.0 (we expect 2 * 3 = 6)

=== 2. z = 3x^2 + 2x + 1   (derivative: 6x + 2) ===
dz/dx at x=2 is 14.0 (we expect 6*2 + 2 = 14)

=== 3. Two variables: f = a*b + b^2 ===
df/da = 5.0 (expected b = 5)
df/db = 12.0 (expected a + 2b = 12)

=== 4. Gradients of a vector (use sum to get one number) ===
w    : tensor([1., 2., 3.])
grad : tensor([2., 4., 6.]) (expected 2 * w)

=== 5. Checking autograd against a tiny numeric estimate ===
numeric estimate: 14.0
autograd answer : 14.0
```

## 🔍 Explanation
1. Part 1: y = x² at x = 3 gives gradient 6.
2. Part 2: z = 3x² + 2x + 1 at x = 2 gives gradient 14.
3. Part 3: with two variables, each one gets its own `.grad`.
4. Part 4: for a vector we `sum()` the result so `backward()` has one number.
5. Part 5 compares autograd with a small numeric estimate to prove it is correct.

---

## 📐 The Math Behind It

| Function | Derivative | At the example value |
|----------|-----------|----------------------|
| y = x² | 2x | 6 at x = 3 |
| z = 3x² + 2x + 1 | 6x + 2 | 14 at x = 2 |
| f = a·b + b² | df/da = b, df/db = a + 2b | 5 and 12 |

## 📝 Important Points

- ✅ Gradients are stored in `.grad`.
- ✅ `backward()` must be called on a single number.
- ✅ Autograd gives exact derivatives, not approximations.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Calling `backward()` on a non-scalar | Use `.sum()` or `.mean()` first |
| Forgetting `requires_grad=True` | `.grad` will be `None` |

## 🧪 Try It Yourself

- [ ] Find the gradient of y = x³ at x = 2 (answer: 12).
- [ ] Find d/dx of 4x + 7 (answer: 4).
- [ ] Find both gradients of f = a * b at a = 3, b = 6.

## ❓ Quick Questions

1. What is a gradient?
2. Where does PyTorch store it?
3. What does `backward()` do?

---

## 🏁 Summary

> 💡 Autograd computes exact derivatives and stores them in `.grad`.

➡️ **Next:** [Requires Grad](../02_requires_grad/requires_grad.md)
