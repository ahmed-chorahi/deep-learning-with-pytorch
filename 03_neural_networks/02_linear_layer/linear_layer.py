# Lesson 2: The linear layer
# output = input @ weight.T + bias

import torch
import torch.nn as nn

torch.manual_seed(0)
layer = nn.Linear(in_features=3, out_features=2)

print("=== 1. Weight and bias ===")
print("weight shape:", tuple(layer.weight.shape))   # (2, 3)
print("bias shape  :", tuple(layer.bias.shape))     # (2,)

print("\n=== 2. One sample ===")
x = torch.tensor([[1.0, 2.0, 3.0]])
out = layer(x)
print("layer output:", out)

print("\n=== 3. Doing the math ourselves ===")
manual = x @ layer.weight.T + layer.bias
print("manual output:", manual)
print("same result?", torch.allclose(out, manual))

print("\n=== 4. A batch of samples ===")
batch = torch.rand(5, 3)
print("input shape :", tuple(batch.shape))
print("output shape:", tuple(layer(batch).shape))

print("\n=== 5. Counting parameters ===")
for in_f, out_f in [(3, 2), (10, 5), (784, 10)]:
    l = nn.Linear(in_f, out_f)
    count = sum(p.numel() for p in l.parameters())
    print(f"Linear({in_f}, {out_f}) -> {in_f}*{out_f} + {out_f} = {count} parameters")

print("\n=== 6. Why we need activation functions ===")
first = nn.Linear(3, 4)
second = nn.Linear(4, 2)
two_layers = second(first(x))
# Two linear layers with nothing in between equal ONE linear layer
combined_weight = second.weight @ first.weight
combined_bias = second.weight @ first.bias + second.bias
one_layer = x @ combined_weight.T + combined_bias
print("two layers == one layer?", torch.allclose(two_layers, one_layer, atol=1e-6))

print("\n=== 7. Setting our own weights ===")
simple = nn.Linear(1, 1)
with torch.no_grad():
    simple.weight.fill_(2.0)
    simple.bias.fill_(1.0)
print("f(5) with w=2, b=1 ->", simple(torch.tensor([[5.0]])).item())
