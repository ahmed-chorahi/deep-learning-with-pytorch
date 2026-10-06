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
