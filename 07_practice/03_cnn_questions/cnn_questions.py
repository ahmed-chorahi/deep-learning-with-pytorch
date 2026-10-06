# CNN questions
import torch
import torch.nn as nn


def conv_output_size(size, kernel, padding=0, stride=1):
    # formula: (size - kernel + 2*padding) / stride + 1
    return (size - kernel + 2 * padding) // stride + 1


# Q1: Output shape of this convolution?
conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)   # no padding
x = torch.rand(1, 3, 32, 32)
print("Q1:", tuple(conv(x).shape), "(32 - 3 + 1 = 30)")

# Q2: How does padding=1 change it?
conv_pad = nn.Conv2d(3, 6, kernel_size=3, padding=1)
print("Q2:", tuple(conv_pad(x).shape), "(size stays the same)")

# Q3: What does MaxPool2d(2) do to the size?
print("Q3:", tuple(nn.MaxPool2d(2)(torch.rand(1, 6, 32, 32)).shape), "(halved)")

# Q4: How many numbers go into the Linear layer after flatten?
out = nn.MaxPool2d(2)(conv_pad(x))
print("Q4:", out.flatten(start_dim=1).shape[1], "= 6 * 16 * 16")

# Q5: Complete a CNN for 28x28 grayscale images with 10 classes
model = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> (4, 14, 14)
    nn.Flatten(),
    nn.Linear(4 * 14 * 14, 10),
)
print("Q5:", tuple(model(torch.rand(2, 1, 28, 28)).shape))

# Q6: Why use a CNN instead of only Linear layers for images?
print("Q6: CNN filters are reused across the image, so they need far fewer parameters")
print("    and can find patterns (like edges) anywhere in the picture.")

# Q7: Check our formula against PyTorch
size = 32
print("\nQ7: size after each step (formula)")
size = conv_output_size(size, kernel=3, padding=1)
print("  conv 3x3, padding 1 ->", size)
size = size // 2
print("  pool 2              ->", size)
size = conv_output_size(size, kernel=5)
print("  conv 5x5, no padding->", size)

# Q8: Count parameters of Conv2d(3, 6, 3):  6 * (3*3*3) weights + 6 biases
count = sum(p.numel() for p in conv.parameters())
print("\nQ8:", count, "= 6*(3*3*3) + 6")
