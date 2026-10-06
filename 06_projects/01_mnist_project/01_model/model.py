# Model for the MNIST digits project
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)        # randomly switches off neurons while training
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))   # (32, 7, 7)
        x = x.flatten(start_dim=1)                # (1568)
        x = self.dropout(self.relu(self.fc1(x)))  # (64)
        return self.fc2(x)                        # (10)


if __name__ == "__main__":
    import torch
    model = Net()
    print(model)
    print("output shape:", tuple(model(torch.rand(2, 1, 28, 28)).shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
