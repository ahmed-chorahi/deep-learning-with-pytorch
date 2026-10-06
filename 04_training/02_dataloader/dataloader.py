# Lesson 2: DataLoader
# A DataLoader hands out the dataset in small groups called batches.

import torch
from torch.utils.data import Dataset, DataLoader


# Same dataset as in 01_dataset/dataset.py (copied so this lesson runs on its own)
class NumbersDataset(Dataset):
    def __init__(self, n=10):
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        self.y = 2 * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


torch.manual_seed(0)
dataset = NumbersDataset(n=10)

print("=== 1. Batches of 4 (no shuffle) ===")
loader = DataLoader(dataset, batch_size=4, shuffle=False)
print("number of batches:", len(loader))      # 10 / 4 -> 3 batches
for i, (x, y) in enumerate(loader):
    print(f"batch {i}: x = {x.flatten().tolist()}  (shape {tuple(x.shape)})")

print("\n=== 2. Shuffling (new order every epoch) ===")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for epoch in range(2):
    order = []
    for x, y in loader:
        order += x.flatten().int().tolist()
    print(f"epoch {epoch}: order = {order}")

print("\n=== 3. drop_last=True throws away the small last batch ===")
loader = DataLoader(dataset, batch_size=4, drop_last=True)
print("number of batches:", len(loader))

print("\n=== 4. Different batch sizes ===")
for size in [1, 2, 5, 10]:
    loader = DataLoader(dataset, batch_size=size)
    print(f"batch_size = {size:2d} -> {len(loader):2d} batches per epoch")

print("\n=== 5. Getting just one batch ===")
x, y = next(iter(DataLoader(dataset, batch_size=3)))
print("x:", x.flatten().tolist(), "| y:", y.flatten().tolist())

print("\n=== 6. A tiny summary after one pass ===")
loader = DataLoader(dataset, batch_size=4)
total = 0
for x, y in loader:
    total += len(x)
print("samples seen in one epoch:", total)
