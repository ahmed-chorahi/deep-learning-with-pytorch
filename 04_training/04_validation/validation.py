# Lesson 4: Validation
# We keep some data hidden from training and use it to check the model honestly.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# Same dataset as in 01_dataset/dataset.py (copied so this lesson runs on its own)
class NumbersDataset(Dataset):
    def __init__(self, n=10, noise=0.0):
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1) / 10     # keep numbers small
        self.y = 2 * self.x + 1 + noise * torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


torch.manual_seed(0)

# Split: 40 samples to train on, 10 samples to validate with
full_data = NumbersDataset(n=50, noise=0.2)
train_data, val_data = random_split(full_data, [40, 10])
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

best_val_loss = float("inf")
best_epoch = 0

print("epoch | train loss | val loss")
for epoch in range(1, 101):
    # ----- training -----
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ----- validation (no learning here) -----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            val_loss += loss_fn(model(x), y).item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:          # remember the best epoch
        best_val_loss = val_loss
        best_epoch = epoch

    if epoch % 20 == 0 or epoch == 1:
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f}")

print()
print(f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch}")
print("Learned weight: %.2f (true 2), bias: %.2f (true 1)" % (model.weight.item(), model.bias.item()))

print("\nIf train loss keeps falling but val loss rises, the model is OVERFITTING.")
