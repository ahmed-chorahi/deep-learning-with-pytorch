# 📈 Backward

> 📚 **Autograd** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
See how `backward()` fills `.grad` and how gradient descent uses it.

## 📌 What is it?
`backward()` runs **backpropagation**: it walks backward through the recorded operations and calculates every gradient. Gradient descent then moves each value a little in the direction that lowers the loss.

## 🧠 Real-Life Analogy
Walking downhill in fog: feel the slope (backward), take a small step down (update), repeat until you reach the bottom.

---

## 💻 Example

Code from `backward.py` (run it with `python backward.py`):

```python
# Lesson 3: backward() and gradient descent

import torch

print("=== 1. backward() fills in .grad ===")
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (w ** 2).sum()
loss.backward()
print("Gradient:", w.grad)                 # 2*w = [2, 4, 6]

print("\n=== 2. Gradients ACCUMULATE ===")
loss = (w ** 2).sum()
loss.backward()
print("After a second backward:", w.grad)  # doubled

w.grad.zero_()                             # reset
print("After zero_():", w.grad)

print("\n=== 3. Gradient descent: minimize (w - 5)^2 ===")
w = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1
for step in range(20):
    loss = (w - 5) ** 2
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad        # step downhill
    w.grad.zero_()
    if step % 5 == 0:
        print(f"step {step:2d}: w = {w.item():.3f}, loss = {loss.item():.3f}")
print("final w (should be near 5):", round(w.item(), 3))

print("\n=== 4. Fit a line y = w*x + b by hand ===")
# the true rule is y = 2x + 1
x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
y = 2 * x + 1

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.02

for epoch in range(300):
    prediction = w * x + b
    loss = ((prediction - y) ** 2).mean()  # mean squared error
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}: loss = {loss.item():.4f}")

print(f"learned w = {w.item():.2f} (true 2), b = {b.item():.2f} (true 1)")
```

## 📤 Expected Output

```text
=== 1. backward() fills in .grad ===
Gradient: tensor([2., 4., 6.])

=== 2. Gradients ACCUMULATE ===
After a second backward: tensor([ 4.,  8., 12.])
After zero_(): tensor([0., 0., 0.])

=== 3. Gradient descent: minimize (w - 5)^2 ===
step  0: w = 1.000, loss = 25.000
step  5: w = 3.689, loss = 2.684
step 10: w = 4.571, loss = 0.288
step 15: w = 4.859, loss = 0.031
final w (should be near 5): 4.942

=== 4. Fit a line y = w*x + b by hand ===
epoch   0: loss = 33.0000
epoch 100: loss = 0.0021
epoch 200: loss = 0.0002
learned w = 2.00 (true 2), b = 0.99 (true 1)
```

## 🔍 Explanation
1. Part 1: `backward()` on the loss fills `w.grad` with `2*w`.
2. Part 2: calling `backward()` again adds to the gradient; `zero_()` resets it.
3. Part 3: gradient descent finds w that minimizes `(w - 5)²`.
4. Part 4: fits the line `y = 2x + 1` by hand, updating `w` and `b` for 300 epochs.
5. This is exactly what an optimizer does later in training.

---

## 📐 The Gradient Descent Recipe

```
repeat:
    loss = f(w)             # forward
    loss.backward()         # compute gradient
    w = w - lr * w.grad     # step downhill
    w.grad.zero_()          # reset
```

`lr` (learning rate) is the size of each step.

## 📝 Important Points

- ✅ Gradients **accumulate**; always reset them between steps.
- ✅ A learning rate that is too big can overshoot; too small is slow.
- ✅ Update weights inside `torch.no_grad()`.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Not resetting gradients | Call `zero_()` / `zero_grad()` every step |
| Updating weights without `no_grad` | Wrap the update in `torch.no_grad()` |

## 🧪 Try It Yourself

- [ ] Change the learning rate to 0.5 and to 0.01 and watch the result.
- [ ] Minimize `(w + 2)²`.
- [ ] Remove `zero_()` and see what happens.

## ❓ Quick Questions

1. Why do we call `zero_grad` / `zero_()`?
2. What is the learning rate?
3. Why is the update done inside `no_grad`?

---

## 🏁 Summary

> 💡 backward() gives gradients; gradient descent uses them to improve the weights.

⬅️ **Previous:** [Requires Grad](../02_requires_grad/requires_grad.md)  |  ➡️ **Next:** [Computational Graph](../04_computational_graph/computational_graph.md)
