# 🏋️ Training Loop

> 📚 **Training** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
Write a complete training loop.

## 📌 What is it?
Training means repeating five steps many times: forward pass, loss, clear gradients, backward pass, update weights. Here a one-neuron model learns the rule `y = 2x + 1`.

## 🧠 Real-Life Analogy
Like learning to throw darts: throw (forward), see how far you missed (loss), adjust your aim (update), repeat.

---

## 💻 Example

Code from `training_loop.py` (run it with `python training_loop.py`):

```python
# Lesson 3: The training loop
# The same 5 steps repeat again and again:
#   forward pass -> loss -> zero_grad -> backward -> step

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Same dataset as in 01_dataset/dataset.py (copied so this lesson runs on its own)
class NumbersDataset(Dataset):
    def __init__(self, n=10, noise=0.0):
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1) / 10   # small numbers train more smoothly
        self.y = 2 * self.x + 1 + noise * torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


torch.manual_seed(0)

data = NumbersDataset(n=20, noise=0.1)
loader = DataLoader(data, batch_size=5, shuffle=True)

model = nn.Linear(1, 1)                           # learns y = w*x + b
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Before training: w = %.2f, b = %.2f" % (model.weight.item(), model.bias.item()))

loss_history = []
for epoch in range(200):                          # one epoch = one pass over the data
    epoch_loss = 0.0
    for x, y in loader:
        prediction = model(x)                     # 1. forward pass
        loss = loss_fn(prediction, y)             # 2. compute the loss
        optimizer.zero_grad()                     # 3. clear old gradients
        loss.backward()                           # 4. backpropagation
        optimizer.step()                          # 5. update the weights
        epoch_loss += loss.item()

    average = epoch_loss / len(loader)
    loss_history.append(average)
    if epoch % 40 == 0:
        bar = "#" * min(int(average * 20), 40)
        print(f"epoch {epoch:3d} | loss = {average:9.4f} | {bar}")

print(f"epoch 199 | loss = {loss_history[-1]:9.4f}")
print()
print("Learned weight: %.3f (true value 2)" % model.weight.item())
print("Learned bias  : %.3f (true value 1)" % model.bias.item())

print("\n=== Predictions on new numbers ===")
model.eval()
with torch.no_grad():
    for number in [1.0, 2.5]:
        answer = model(torch.tensor([[number]])).item()
        print(f"x = {number:4.1f} -> predicted {answer:6.2f} (true {2 * number + 1:.1f})")

best = min(loss_history)
print("\nlowest loss:", round(best, 4), "at epoch", loss_history.index(best))
```

## 📤 Expected Output

```text
Before training: w = -0.58, b = 0.86
epoch   0 | loss =    3.9612 | ########################################
epoch  40 | loss =    0.0083 | 
epoch  80 | loss =    0.0083 | 
epoch 120 | loss =    0.0078 | 
epoch 160 | loss =    0.0092 | 
epoch 199 | loss =    0.0076

Learned weight: 2.105 (true value 2)
Learned bias  : 0.941 (true value 1)

=== Predictions on new numbers ===
x =  1.0 -> predicted   3.05 (true 3.0)
x =  2.5 -> predicted   6.20 (true 6.0)

lowest loss: 0.0075 at epoch 188
```

## 🔍 Explanation
1. Creates a noisy dataset (`y ≈ 2x + 1`) and a DataLoader.
2. Model: `nn.Linear(1, 1)`; loss: MSE; optimizer: SGD.
3. The five-step loop runs for 200 epochs.
4. The average loss per epoch is stored and printed with a small bar.
5. Prints learned weight and bias (close to 2 and 1).
6. Predicts new numbers and finds the epoch with the lowest loss.

---

## 📐 The 5 Steps

```
1. prediction = model(x)
2. loss = loss_fn(prediction, y)
3. optimizer.zero_grad()
4. loss.backward()
5. optimizer.step()
```

**Epoch** = one pass over the data. **Batch** = a small group of samples.

## 📝 Important Points

- ✅ The loss should go down over epochs.
- ✅ Always call `zero_grad()` before `backward()`.
- ✅ The learning rate controls step size.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting `optimizer.zero_grad()` | Gradients would add up |
| Learning rate too high | The loss explodes; try a smaller value |

## 🧪 Try It Yourself

- [ ] Change `lr` to 0.001 and see how the loss changes.
- [ ] Train for 500 epochs.
- [ ] Change the rule in `dataset` to `y = 3x` and see if the model learns it.

## ❓ Quick Questions

1. What are the 5 steps of training?
2. What is an epoch?
3. What does the optimizer do?

---

## 🏁 Summary

> 💡 Training = forward, loss, zero_grad, backward, step, repeated for many epochs.

⬅️ **Previous:** [Dataloader](../02_dataloader/dataloader.md)  |  ➡️ **Next:** [Validation](../04_validation/validation.md)
