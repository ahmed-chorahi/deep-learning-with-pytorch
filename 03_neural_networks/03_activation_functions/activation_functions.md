# 🧠 Activation Functions

> 📚 **Neural Networks** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
Know the common activation functions and what they do.

## 📌 What is it?
An **activation function** is applied after a layer to add non-linearity. Without them, a network could only learn straight-line relationships.

## 🧠 Real-Life Analogy
Think of a light dimmer vs. a switch. Activation functions decide how strongly each neuron 'fires'.

---

## 💻 Example

Code from `activation_functions.py` (run it with `python activation_functions.py`):

```python
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
```

## 📤 Expected Output

```text
input:      tensor([-2., -1.,  0.,  1.,  2.])

=== 1. The main activation functions ===
ReLU:       tensor([0., 0., 0., 1., 2.])
LeakyReLU:  tensor([-0.2000, -0.1000,  0.0000,  1.0000,  2.0000])
Sigmoid:    tensor([0.1190, 0.2690, 0.5000, 0.7310, 0.8810])
Tanh:       tensor([-0.9640, -0.7620,  0.0000,  0.7620,  0.9640])

=== 2. Softmax turns scores into probabilities ===
scores: tensor([2.0000, 1.0000, 0.1000])
probs : tensor([0.6590, 0.2420, 0.0990])
sum   : 1.0000001192092896

=== 3. Text picture of ReLU and Sigmoid ===
x = -4 | ReLU      | Sigmoid            0.02
x = -3 | ReLU      | Sigmoid            0.05
x = -2 | ReLU      | Sigmoid #          0.12
x = -1 | ReLU      | Sigmoid ##         0.27
x =  0 | ReLU      | Sigmoid #####      0.50
x =  1 | ReLU #    | Sigmoid #######    0.73
x =  2 | ReLU ##   | Sigmoid ########   0.88
x =  3 | ReLU ###  | Sigmoid #########  0.95
x =  4 | ReLU #### | Sigmoid #########  0.98

=== 4. Using an activation inside a model ===
Sequential(
  (0): Linear(in_features=2, out_features=4, bias=True)
  (1): ReLU()
  (2): Linear(in_features=4, out_features=1, bias=True)
  (3): Sigmoid()
)
output for one sample: tensor([[0.5172]], grad_fn=<SigmoidBackward0>)

=== 5. How many neurons does ReLU switch off? ===
fraction of zeros: 0.49 (about half)
```

## 🔍 Explanation
1. Applies ReLU, LeakyReLU, Sigmoid and Tanh to the same numbers.
2. Softmax turns scores into probabilities that sum to 1.
3. A text chart compares ReLU and Sigmoid for x from -4 to 4.
4. Activations are placed inside `nn.Sequential`.
5. ReLU switches off about half of random neurons.

---

## 📐 Output Ranges

| Function | Range | Use |
|----------|-------|-----|
| ReLU | 0 to ∞ | hidden layers |
| Sigmoid | 0 to 1 | yes/no probability |
| Tanh | -1 to 1 | hidden layers |
| Softmax | 0 to 1 (sum = 1) | multi-class probabilities |

## 📝 Important Points

- ✅ ReLU is the default choice for hidden layers.
- ✅ Softmax needs a `dim` argument.
- ✅ `CrossEntropyLoss` already includes softmax, so don't add it before the loss.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Using Sigmoid in every hidden layer | ReLU is the usual choice |
| Applying softmax before `CrossEntropyLoss` | The loss already includes softmax |

## 🧪 Try It Yourself

- [ ] Apply ReLU to `[-5, 0, 5]`.
- [ ] Check that softmax of `[1, 1, 1]` gives equal probabilities.
- [ ] Which input gives sigmoid = 0.5? (answer: 0)

## ❓ Quick Questions

1. Why do we need activation functions?
2. What does ReLU do to negatives?
3. What do softmax outputs add up to?

---

## 🏁 Summary

> 💡 Activations add curves so networks can learn more than straight lines.

⬅️ **Previous:** [Linear Layer](../02_linear_layer/linear_layer.md)  |  ➡️ **Next:** [Loss Functions](../04_loss_functions/loss_functions.md)
