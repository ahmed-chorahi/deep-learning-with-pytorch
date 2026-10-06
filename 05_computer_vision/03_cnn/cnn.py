# Lesson 3: A simple CNN (Convolutional Neural Network)
# Small filters slide over the image to find patterns like edges and curves.

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)    # (1,28,28) -> (8,28,28)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)   # (8,14,14) -> (16,14,14)
        self.pool = nn.MaxPool2d(2)                               # halves height and width
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 7 * 7, 10)                       # 10 classes

    def forward(self, x, show_shapes=False):
        x = self.pool(self.relu(self.conv1(x)))
        if show_shapes:
            print("after conv1 + pool:", tuple(x.shape))
        x = self.pool(self.relu(self.conv2(x)))
        if show_shapes:
            print("after conv2 + pool:", tuple(x.shape))
        x = x.flatten(start_dim=1)
        if show_shapes:
            print("after flatten     :", tuple(x.shape))
        return self.fc(x)


if __name__ == "__main__":
    model = SimpleCNN()
    print("=== 1. The model ===")
    print(model)

    print("\n=== 2. Shapes through the network ===")
    fake_images = torch.rand(4, 1, 28, 28)         # 4 random "images"
    print("input             :", tuple(fake_images.shape))
    output = model(fake_images, show_shapes=True)
    print("output            :", tuple(output.shape))

    print("\n=== 3. Parameters ===")
    for name, p in model.named_parameters():
        print(f"{name:14s} {str(tuple(p.shape)):18s} {p.numel()}")
    cnn_total = sum(p.numel() for p in model.parameters())
    print("total:", cnn_total)

    print("\n=== 4. CNN vs plain linear layers ===")
    plain = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    plain_total = sum(p.numel() for p in plain.parameters())
    print("plain network parameters:", plain_total)
    print("our CNN parameters      :", cnn_total)

    print("\n=== 5. One filter up close ===")
    print("conv1 weight shape:", tuple(model.conv1.weight.shape), "= (filters, channels, height, width)")
    print("first filter:\n", model.conv1.weight[0, 0].detach().round(decimals=2))
