# Lesson 3: Activation functions
# They add non-linearity so a network can learn curves, not only straight lines.

import torch
import torch.nn as nn

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("input:     ", x)

print("\n=== 1. The main activation functions ===")
print("ReLU:      ", nn.ReLU()(x))              # negatives become 0
print("LeakyReLU: ", nn.LeakyReLU(0.1)(x))      # negatives shrink a bit
print("Sigmoid:   ", nn.Sigmoid()(x).round(decimals=3))
print("Tanh:      ", nn.Tanh()(x).round(decimals=3))

print("\n=== 2. Softmax turns scores into probabilities ===")
scores = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(scores, dim=0)
print("scores:", scores)
print("probs :", probs.round(decimals=3))
print("sum   :", probs.sum().item())

print("\n=== 3. Text picture of ReLU and Sigmoid ===")
for value in range(-4, 5):
    v = torch.tensor(float(value))
    relu = torch.relu(v).item()
    sig = torch.sigmoid(v).item()
    print(f"x = {value:2d} | ReLU {'#' * int(relu):<4s} | Sigmoid {'#' * int(sig * 10):<10s} {sig:.2f}")

print("\n=== 4. Using an activation inside a model ===")
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)
print(model)
print("output for one sample:", model(torch.tensor([[1.0, -1.0]])))

print("\n=== 5. How many neurons does ReLU switch off? ===")
torch.manual_seed(1)
hidden = torch.randn(1000)
after = torch.relu(hidden)
print("fraction of zeros:", round((after == 0).float().mean().item(), 2), "(about half)")
