# Lesson 1: MNIST
# 70,000 handwritten digits (0-9), each a 28x28 grayscale image.
# The first run downloads the data into a folder called "data" (needs internet).

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ToTensor turns a picture into a tensor with values between 0 and 1
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

print("=== 1. Dataset size ===")
print("Training images:", len(train_data))
print("Test images:", len(test_data))

print("\n=== 2. One image ===")
image, label = train_data[0]
print("One image shape:", image.shape)        # (1, 28, 28) = channels, height, width
print("Its label:", label)
print("Pixel range:", image.min().item(), "to", image.max().item())

print("\n=== 3. Drawing the digit with text ===")
for row in image[0][::2]:                     # every 2nd row so it fits on screen
    print("".join("#" if p > 0.5 else "." for p in row))

print("\n=== 4. How many of each digit? (first 1000 images) ===")
counts = [0] * 10
for i in range(1000):
    counts[train_data[i][1]] += 1
for digit, count in enumerate(counts):
    print(f"digit {digit}: {count:3d} {'#' * (count // 5)}")

print("\n=== 5. Batches ===")
loader = DataLoader(train_data, batch_size=64, shuffle=True)
images, labels = next(iter(loader))
print("Batch of images:", images.shape)       # (64, 1, 28, 28)
print("Batch of labels:", labels.shape)       # (64,)
print("First 10 labels:", labels[:10].tolist())

print("\n=== 6. Average pixel value ===")
print("mean pixel of this batch:", round(images.mean().item(), 3))
