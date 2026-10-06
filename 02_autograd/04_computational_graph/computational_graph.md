# 📈 Computational Graph

> 📚 **Autograd** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Understand how PyTorch records operations as a graph.

## 📌 What is it?
Every operation on a tracked tensor is recorded in a **computational graph**. The graph remembers how each result was made so `backward()` can go in reverse and apply the chain rule.

## 🧠 Real-Life Analogy
It's like a recipe card kept while cooking. If you want to know how the final dish changes when you add more salt, you read the recipe backwards.

---

## 💻 Example

Code from `computational_graph.py` (run it with `python computational_graph.py`):

```python
# Lesson 4: The computational graph
# PyTorch remembers every operation so it can go backwards later.

import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3          # step 1
z = y + 4          # step 2
out = z ** 2       # step 3  ->  out = (3x + 4)^2

print("=== 1. grad_fn: how was each tensor made? ===")
print("x  :", x.grad_fn)
print("y  :", y.grad_fn)
print("z  :", z.grad_fn)
print("out:", out.grad_fn)

print("\n=== 2. Leaf tensors ===")
for name, t in [("x", x), ("y", y), ("z", z), ("out", out)]:
    print(f"{name:3s} is_leaf = {t.is_leaf}")

print("\n=== 3. Walking backwards through the graph ===")
node = out.grad_fn
step = 1
while node is not None:
    print(f"step {step}: {node.name()}")
    if len(node.next_functions) == 0:
        break
    node = node.next_functions[0][0]
    step += 1

print("\n=== 4. Backward pass ===")
out.backward()
print("d(out)/dx =", x.grad.item(), "(chain rule: 2*(3x+4)*3 = 60)")

print("\n=== 5. The graph is freed after backward() ===")
try:
    out.backward()
except RuntimeError as error:
    print("Second backward failed:", str(error)[:60], "...")

print("\n=== 6. A new forward pass builds a new graph ===")
x.grad = None
out = (3 * x + 4) ** 2
out.backward()
print("gradient again:", x.grad.item())

print("\n=== 7. detach() cuts the graph ===")
y2 = (x * 3).detach()
print("y2.requires_grad:", y2.requires_grad)
```

## 📤 Expected Output

```text
=== 1. grad_fn: how was each tensor made? ===
x  : None
y  : <MulBackward0 object at 0x7f0ebec38490>
z  : <AddBackward0 object at 0x7f0ebec38c10>
out: <PowBackward0 object at 0x7f0ebec38be0>

=== 2. Leaf tensors ===
x   is_leaf = True
y   is_leaf = False
z   is_leaf = False
out is_leaf = False

=== 3. Walking backwards through the graph ===
step 1: PowBackward0
step 2: AddBackward0
step 3: MulBackward0
step 4: torch::autograd::AccumulateGrad

=== 4. Backward pass ===
d(out)/dx = 60.0 (chain rule: 2*(3x+4)*3 = 60)

=== 5. The graph is freed after backward() ===
Second backward failed: Trying to backward through the graph a second time (or direc ...

=== 6. A new forward pass builds a new graph ===
gradient again: 60.0

=== 7. detach() cuts the graph ===
y2.requires_grad: False
```

## 🔍 Explanation
1. We build `out = (3x + 4)²` in three steps.
2. `grad_fn` names the operation that created each tensor.
3. `is_leaf` is True only for tensors we created.
4. A `while` loop walks backwards through `next_functions`.
5. `backward()` gives 60; calling it twice raises an error because the graph was freed.
6. A new forward pass builds a new graph; `detach()` cuts the graph.

---

## 📐 The Graph in This Example

```
x --(*3)--> y --(+4)--> z --(**2)--> out
```

Going backward: d(out)/dz = 2z, dz/dy = 1, dy/dx = 3. Multiply them together (the chain rule).

## 📝 Important Points

- ✅ `grad_fn` shows which operation made a tensor.
- ✅ Leaf tensors have `grad_fn = None`.
- ✅ A new graph is built on every forward pass.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Calling `backward()` twice on the same graph | Run the forward pass again |
| Expecting `.grad` on non-leaf tensors | Only leaf tensors keep `.grad` |

## 🧪 Try It Yourself

- [ ] Add one more operation and print its `grad_fn`.
- [ ] Predict d(out)/dx at x = 1, then check.
- [ ] Try calling `out.backward()` twice and read the error.

## ❓ Quick Questions

1. What is a computational graph?
2. What is a leaf tensor?
3. Why does PyTorch free the graph after `backward()`?

---

## 🏁 Summary

> 💡 PyTorch records operations in a graph and replays it backwards to get gradients.

⬅️ **Previous:** [Backward](../03_backward/backward.md)  |  ➡️ **Next:** [Practice](../05_practice/practice.md)
