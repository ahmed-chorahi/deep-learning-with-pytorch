# Lesson 1: nn.Module
# Every PyTorch model is a class that inherits from nn.Module.

import torch
import torch.nn as nn


class MyFirstNet(nn.Module):
    def __init__(self):
        super().__init__()                 # always call this first
        self.layer = nn.Linear(3, 2)       # 3 inputs -> 2 outputs

    def forward(self, x):
        # describes how data flows through the model
        return self.layer(x)


model = MyFirstNet()
print("=== 1. The model ===")
print(model)

print("\n=== 2. Running data through it ===")
x = torch.rand(4, 3)                       # 4 samples, 3 features each
output = model(x)                          # this calls forward()
print("input shape :", tuple(x.shape))
print("output shape:", tuple(output.shape))

print("\n=== 3. Parameters ===")
for name, param in model.named_parameters():
    print(f"{name:13s} shape = {tuple(param.shape)}")
total = sum(p.numel() for p in model.parameters())
print("total parameters:", total)          # 3*2 weights + 2 biases = 8

print("\n=== 4. A model with two layers ===")


class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


bigger = TwoLayerNet()
print(bigger)
print("output shape:", tuple(bigger(x).shape))
print("total parameters:", sum(p.numel() for p in bigger.parameters()))

print("\n=== 5. state_dict keys (what gets saved) ===")
for key in bigger.state_dict().keys():
    print(" ", key)

print("\n=== 6. train() and eval() modes ===")
bigger.train()
print("training mode:", bigger.training)
bigger.eval()
print("training mode:", bigger.training)
