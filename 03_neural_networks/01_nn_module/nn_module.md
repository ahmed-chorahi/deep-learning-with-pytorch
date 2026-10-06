# 🧠 Nn Module

> 📚 **Neural Networks** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Learn how to build a network using `nn.Module`.

## 📌 What is it?
`nn.Module` is the base class for every PyTorch model. You define layers in `__init__` and describe how data flows through them in `forward`.

## 🧠 Real-Life Analogy
A module is like a factory blueprint: `__init__` lists the machines, `forward` says the order the product passes through them.

---

## 💻 Example

Code from `nn_module.py` (run it with `python nn_module.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. The model ===
MyFirstNet(
  (layer): Linear(in_features=3, out_features=2, bias=True)
)

=== 2. Running data through it ===
input shape : (4, 3)
output shape: (4, 2)

=== 3. Parameters ===
layer.weight  shape = (2, 3)
layer.bias    shape = (2,)
total parameters: 8

=== 4. A model with two layers ===
TwoLayerNet(
  (hidden): Linear(in_features=3, out_features=5, bias=True)
  (relu): ReLU()
  (output): Linear(in_features=5, out_features=2, bias=True)
)
output shape: (4, 2)
total parameters: 32

=== 5. state_dict keys (what gets saved) ===
  hidden.weight
  hidden.bias
  output.weight
  output.bias

=== 6. train() and eval() modes ===
training mode: True
training mode: False

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. `MyFirstNet` inherits from `nn.Module`; `__init__` makes a layer and `forward` uses it.
2. `model(x)` calls `forward` automatically.
3. `named_parameters()` shows weights and biases (8 in total).
4. `TwoLayerNet` adds a hidden layer and ReLU.
5. `state_dict().keys()` lists what gets saved.
6. `train()` and `eval()` switch modes.

---

## 📐 Shapes

```
input  : (4, 3)   4 samples, 3 features
layer  : Linear(3 -> 2)
output : (4, 2)
```

Parameters: 3×2 weights + 2 biases = **8**.

## 📝 Important Points

- ✅ Always call `super().__init__()` first.
- ✅ Call `model(x)`, not `model.forward(x)`.
- ✅ Parameters are found automatically from the layers you create.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting `super().__init__()` | It must be the first line |
| Calling `model.forward(x)` | Call `model(x)` |

## 🧪 Try It Yourself

- [ ] Change the layer to `nn.Linear(3, 5)` and count the parameters (20).
- [ ] Add a second layer and update `forward`.
- [ ] Print `model.layer.weight`.

## ❓ Quick Questions

1. What does `forward` do?
2. Why call `super().__init__()`?
3. How many parameters does `Linear(3, 2)` have?

---

## 🏁 Summary

> 💡 A model is a class with layers in `__init__` and the data flow in `forward`.

➡️ **Next:** [Linear Layer](../02_linear_layer/linear_layer.md)
