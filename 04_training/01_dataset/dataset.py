# Lesson 1: Dataset
# A Dataset tells PyTorch (1) how many samples we have and (2) how to get one sample.

import torch
from torch.utils.data import Dataset


class NumbersDataset(Dataset):
    """A tiny dataset where the answer follows y = 2x + 1 (plus optional noise)."""

    def __init__(self, n=10, noise=0.0):
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1)   # shape (n, 1)
        self.y = 2 * self.x + 1
        if noise > 0:
            self.y = self.y + noise * torch.randn(n, 1)              # make it a bit messy

    def __len__(self):
        return len(self.x)                    # how many samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]   # one (input, target) pair


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== 1. Clean dataset ===")
    dataset = NumbersDataset(n=5)
    print("size:", len(dataset))
    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"sample {i}: x = {x.item():.1f}, y = {y.item():.1f}")

    print("\n=== 2. Noisy dataset ===")
    noisy = NumbersDataset(n=5, noise=0.5)
    for i in range(len(noisy)):
        x, y = noisy[i]
        print(f"sample {i}: x = {x.item():.1f}, y = {y.item():.2f}")

    print("\n=== 3. Shapes ===")
    x, y = dataset[0]
    print("one x shape:", tuple(x.shape), "| one y shape:", tuple(y.shape))

    print("\n=== 4. Last sample using a negative index ===")
    print(dataset[-1])

    print("\n=== 5. Splitting into train and test ===")
    full = NumbersDataset(n=10)
    train_set, test_set = torch.utils.data.random_split(full, [8, 2])
    print("train size:", len(train_set), "| test size:", len(test_set))
