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
