# 📈 Requires Grad

> 📚 **Autograd** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Learn how to turn gradient tracking on and off.

## 📌 What is it?
`requires_grad` is a switch on every tensor. When it is on, PyTorch records every operation so it can calculate gradients later. When it is off, PyTorch saves memory and time.

## 🧠 Real-Life Analogy
It's like a security camera. When it's on, everything is recorded for review later. Turn it off when you don't need a replay.

---

## 💻 Example

Code from `requires_grad.py` (run it with `python requires_grad.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. Turning tracking on ===
a.requires_grad: False
b.requires_grad: True
a.requires_grad now: True

=== 2. Results of tracked tensors are tracked ===
c.requires_grad: True
c.grad_fn: <MulBackward0 object at 0x7f1ee991c0a0>

=== 3. torch.no_grad() for predictions ===
d.requires_grad inside no_grad: False

=== 4. detach() ===
e.requires_grad: False
e as numpy: [2. 4.]

=== 5. Model weights have requires_grad=True ===
weight  requires_grad = True
bias    requires_grad = True

=== 6. Freezing a layer (turn learning off) ===
output requires_grad: False

=== 7. Trying to use grad on a non-float tensor ===
Error: Only Tensors of floating point and complex dtype can require gradients

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. Normal tensors start untracked; `requires_grad=True` or `requires_grad_(True)` switches tracking on.
2. Results of tracked tensors are tracked and get a `grad_fn`.
3. `torch.no_grad()` stops tracking temporarily.
4. `detach()` cuts a tensor from the graph so it can become NumPy.
5. Model weights (`nn.Linear`) already have `requires_grad=True`.
6. Setting `p.requires_grad = False` freezes a layer.
7. Integer tensors cannot require gradients (error caught).

---

## 📐 When to Use Each

| Tool | Typical use |
|------|-------------|
| `requires_grad=True` | model weights you want to learn |
| `torch.no_grad()` | making predictions / validation |
| `.detach()` | converting a result to NumPy or plotting |

## 📝 Important Points

- ✅ Model weights always have `requires_grad=True`.
- ✅ Use `torch.no_grad()` when you only want predictions.
- ✅ Only floating-point tensors can require gradients.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Calling `.numpy()` on a tracked tensor | Use `.detach().numpy()` |
| Leaving tracking on while predicting | Wrap predictions in `torch.no_grad()` |

## 🧪 Try It Yourself

- [ ] Make a tensor with tracking on, multiply by 3, and check `requires_grad`.
- [ ] Do the same inside `torch.no_grad()`.
- [ ] Call `.detach()` and check again.

## ❓ Quick Questions

1. What does `requires_grad=True` do?
2. Why use `torch.no_grad()` for predictions?
3. What does `.detach()` do?

---

## 🏁 Summary

> 💡 `requires_grad` decides what PyTorch learns; `no_grad` and `detach` turn it off.

⬅️ **Previous:** [Gradients](../01_gradients/gradients.md)  |  ➡️ **Next:** [Backward](../03_backward/backward.md)
