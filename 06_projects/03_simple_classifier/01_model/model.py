# Model for a simple 2D point classifier
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),    # 2 inputs: the x and y position of a point
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),    # 2 classes: 0 = outside the circle, 1 = inside
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    import torch
    model = Classifier()
    print(model)
    print("output shape:", tuple(model(torch.rand(5, 2)).shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
