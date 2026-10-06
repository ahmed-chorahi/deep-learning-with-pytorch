# Lesson 3: Building a CNN
# Pattern: [Conv -> ReLU -> Pool] repeated, then Flatten -> Linear.

import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, num_classes=3, image_size=12):
        super().__init__()
        # feature extractor: finds patterns in the image
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # work out how many numbers come out of the feature extractor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            flat_size = self.features(dummy).flatten(start_dim=1).shape[1]
        # classifier: makes the final decision
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = MyCNN()
print("=== 1. The model ===")
print(model)

print("\n=== 2. Shape after every layer ===")
x = torch.rand(4, 1, 12, 12)                  # 4 images, 1 channel, 12x12
print(f"{'input':28s} {tuple(x.shape)}")
for layer in list(model.features) + list(model.classifier):
    x = layer(x)
    print(f"{layer.__class__.__name__:12s} {'':15s} {tuple(x.shape)}")

print("\n=== 3. Parameters per layer ===")
total = 0
for name, p in model.named_parameters():
    total += p.numel()
    print(f"{name:22s} {str(tuple(p.shape)):16s} {p.numel():5d}")
print("total:", total)

print("\n=== 4. Different image size ===")
bigger = MyCNN(num_classes=10, image_size=28)
print("28x28 model output:", tuple(bigger(torch.rand(2, 1, 28, 28)).shape))
print("28x28 model parameters:", sum(p.numel() for p in bigger.parameters()))
