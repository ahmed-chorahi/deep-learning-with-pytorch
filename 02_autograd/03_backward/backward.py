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
