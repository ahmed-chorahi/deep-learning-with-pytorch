# Lesson 2: Fashion-MNIST
# Same format as MNIST (28x28 grayscale, 10 classes), but the pictures are clothes.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print("=== 1. Dataset size ===")
print("Training images:", len(train_data))
print("Test images:", len(test_data))

print("\n=== 2. One image ===")
image, label = train_data[0]
print("Image shape:", image.shape)
print("Label:", label, "->", class_names[label])

print("\n=== 3. Draw it with text ===")
for row in image[0][::2]:
    print("".join("#" if p > 0.5 else "." for p in row))

print("\n=== 4. Class names ===")
for number, name in enumerate(class_names):
    print(f"{number} = {name}")

print("\n=== 5. A batch ===")
loader = DataLoader(train_data, batch_size=32, shuffle=True)
images, labels = next(iter(loader))
print("Batch shape:", images.shape)
print("First 8 items:", [class_names[i] for i in labels[:8]])

print("\n=== 6. Count each class in this batch ===")
for number, name in enumerate(class_names):
    count = (labels == number).sum().item()
    print(f"{name:12s} {count:2d} {'#' * count}")
