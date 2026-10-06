# Lesson 5: BatchNorm and Dropout
# Two layers that help a CNN train faster and overfit less.

import torch
import torch.nn as nn

torch.manual_seed(0)

print("=== 1. BatchNorm: keeps numbers in a healthy range ===")
x = torch.rand(16, 4, 8, 8) * 10 + 5             # numbers around 5 to 15
print(f"before: mean = {x.mean().item():.2f}, std = {x.std().item():.2f}")
bn = nn.BatchNorm2d(4)
bn.train()
y = bn(x)
print(f"after : mean = {y.mean().item():.2f}, std = {y.std().item():.2f}")

print("\n=== 2. Dropout: randomly switches neurons off while training ===")
drop = nn.Dropout(p=0.5)
data = torch.ones(10)
drop.train()
print("train mode:", drop(data))                  # about half are 0, others scaled to 2
drop.eval()
print("eval mode :", drop(data))                  # nothing is dropped

print("\n=== 3. A CNN with both layers ===")
model_with = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Dropout(0.3),
    nn.Linear(16 * 3 * 3, 3),
)
print(model_with)
print("output shape:", tuple(model_with(torch.rand(2, 1, 12, 12)).shape))

print("\n=== 4. Compare with and without (same data, same epochs) ===")
SIZE = 12


def make_dataset(n):
    labels = torch.randint(0, 3, (n,))
    images = torch.zeros(n, 1, SIZE, SIZE)
    for i, label in enumerate(labels):
        pos = torch.randint(2, SIZE - 2, (1,)).item()
        if label == 0:
            images[i, 0, pos, :] = 1
        elif label == 1:
            images[i, 0, :, pos] = 1
        else:
            for k in range(SIZE):
                images[i, 0, k, k] = 1
    return images + 0.4 * torch.rand_like(images), labels


x_train, y_train = make_dataset(300)
x_test, y_test = make_dataset(150)


def train_and_test(model, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), 50):
            idx = order[start:start + 50]
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_acc = (model(x_train).argmax(dim=1) == y_train).float().mean().item() * 100
        test_acc = (model(x_test).argmax(dim=1) == y_test).float().mean().item() * 100
    print(f"{name:22s} train accuracy = {train_acc:5.1f}% | test accuracy = {test_acc:5.1f}%")


plain = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 3 * 3, 3),
)
fancy = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Dropout(0.3), nn.Linear(16 * 3 * 3, 3),
)
train_and_test(plain, "plain CNN")
train_and_test(fancy, "CNN + BatchNorm + Dropout")
