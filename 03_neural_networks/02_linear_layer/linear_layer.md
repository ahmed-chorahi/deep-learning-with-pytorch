# 🧠 Linear Layer

> 📚 **Neural Networks** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Understand what `nn.Linear` calculates.

## 📌 What is it?
A **linear layer** multiplies the input by a weight matrix and adds a bias: `output = input @ weight.T + bias`. It is the basic building block of neural networks.

## 🧠 Real-Life Analogy
Like a recipe that mixes ingredients: each output is a weighted mix of all inputs, plus a constant.

---

## 💻 Example

Code from `linear_layer.py` (run it with `python linear_layer.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. Weight and bias ===
weight shape: (2, 3)
bias shape  : (2,)

=== 2. One sample ===
layer output: tensor([[-0.8219,  0.0526]], grad_fn=<AddmmBackward0>)

=== 3. Doing the math ourselves ===
manual output: tensor([[-0.8219,  0.0526]], grad_fn=<AddBackward0>)
same result? True

=== 4. A batch of samples ===
input shape : (5, 3)
output shape: (5, 2)

=== 5. Counting parameters ===
Linear(3, 2) -> 3*2 + 2 = 8 parameters
Linear(10, 5) -> 10*5 + 5 = 55 parameters
Linear(784, 10) -> 784*10 + 10 = 7850 parameters

=== 6. Why we need activation functions ===
two layers == one layer? True

=== 7. Setting our own weights ===
f(5) with w=2, b=1 -> 11.0
```

## 🔍 Explanation
1. Looks at the weight `(2, 3)` and bias `(2,)` of `Linear(3, 2)`.
2. Runs one sample, then repeats the maths with `x @ weight.T + bias`.
3. Passes a batch of 5 samples.
4. Counts parameters for several layer sizes.
5. Proves two linear layers equal one linear layer (so we need activations).
6. Sets weights by hand so `f(5) = 2*5 + 1 = 11`.

---

## 📐 Shapes

```
x       : (1, 3)
weight  : (2, 3)   -> weight.T is (3, 2)
x @ W.T : (1, 2)
+ bias  : (2,)     (added to each row)
```

## 📝 Important Points

- ✅ Weights start random, so your numbers will differ.
- ✅ Weights and biases are learned during training.
- ✅ Add activation functions between linear layers.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Wrong input size | The last dimension must equal `in_features` |
| Stacking linear layers with no activation | Put ReLU etc. in between |

## 🧪 Try It Yourself

- [ ] Make `nn.Linear(4, 1)` and print its weight shape.
- [ ] Check the manual formula on your own input.
- [ ] Count parameters of `Linear(10, 5)`.

## ❓ Quick Questions

1. What is the formula for a linear layer?
2. What shape is the weight of `Linear(3, 2)`?
3. Why do we need activations between linear layers?

---

## 🏁 Summary

> 💡 A linear layer is `x @ W.T + b`; it learns the weights `W` and bias `b`.

⬅️ **Previous:** [Nn Module](../01_nn_module/nn_module.md)  |  ➡️ **Next:** [Activation Functions](../03_activation_functions/activation_functions.md)
