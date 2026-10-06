# 🧠 Simple Network

> 📚 **Neural Networks** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Build a complete small network: Linear → ReLU → Linear.

## 📌 What is it?
This lesson combines layers, activation and loss into a real (tiny) neural network. It takes 4 numbers and decides between 3 classes.

## 🧠 Real-Life Analogy
Like an assembly line: first station mixes the inputs, the second decides what to keep, the third produces the final scores.

---

## 💻 Example

Code from `simple_network.py` (run it with `python simple_network.py`):

```python
# Lesson 5: A simple neural network
# Linear -> ReLU -> Linear. It takes 4 numbers and picks one of 3 classes.

import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)      # input  -> hidden
        x = self.relu(x)     # non-linearity
        x = self.fc2(x)      # hidden -> output scores
        return x


model = SimpleNetwork(input_size=4, hidden_size=8, output_size=3)
print("=== 1. The model ===")
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

print("\n=== 2. Forward pass with random data ===")
x = torch.rand(5, 4)                       # 5 samples, 4 features
scores = model(x)
print("output shape:", tuple(scores.shape))
print("predicted classes:", scores.argmax(dim=1))

print("\n=== 3. Loss for random weights ===")
labels = torch.tensor([0, 1, 2, 1, 0])
loss_fn = nn.CrossEntropyLoss()
print("loss:", round(loss_fn(scores, labels).item(), 4))

print("\n=== 4. Teaching it on these 5 samples ===")
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for epoch in range(60):
    loss = loss_fn(model(x), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 15 == 0:
        print(f"epoch {epoch:2d} | loss = {loss.item():.4f}")

print("\n=== 5. After training ===")
predictions = model(x).argmax(dim=1)
print("true labels     :", labels.tolist())
print("predicted labels:", predictions.tolist())
accuracy = (predictions == labels).float().mean().item() * 100
print(f"accuracy on these samples: {accuracy:.0f}%")
print("(it memorized 5 samples; real projects use separate test data)")
```

## 📤 Expected Output

```text
=== 1. The model ===
SimpleNetwork(
  (fc1): Linear(in_features=4, out_features=8, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=8, out_features=3, bias=True)
)
parameters: 67

=== 2. Forward pass with random data ===
output shape: (5, 3)
predicted classes: tensor([0, 2, 2, 0, 0])

=== 3. Loss for random weights ===
loss: 1.0847

=== 4. Teaching it on these 5 samples ===
epoch  0 | loss = 1.0847
epoch 15 | loss = 0.3894
epoch 30 | loss = 0.0411
epoch 45 | loss = 0.0078

=== 5. After training ===
true labels     : [0, 1, 2, 1, 0]
predicted labels: [0, 1, 2, 1, 0]
accuracy on these samples: 100%
(it memorized 5 samples; real projects use separate test data)
```

## 🔍 Explanation
1. `SimpleNetwork(4, 8, 3)` is Linear → ReLU → Linear.
2. A forward pass on 5 random samples gives scores of shape `(5, 3)`.
3. `CrossEntropyLoss` measures the error.
4. Adam trains the network for 60 epochs on the 5 samples.
5. After training the predictions match the labels (it memorized them).

---

## 📐 Shapes Through the Network

```
input   (5, 4)
fc1  -> (5, 8)
ReLU -> (5, 8)
fc2  -> (5, 3)
```

## 📝 Important Points

- ✅ Weights are untrained, so predictions are random for now.
- ✅ Hidden size is a free choice; bigger means more capacity.
- ✅ The output size equals the number of classes.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Testing on training data and calling it accuracy | Always keep separate test data |
| Output size not equal to number of classes | Match it to the classes |

## 🧪 Try It Yourself

- [ ] Change the hidden size to 16 and print the parameter count.
- [ ] Add a second hidden layer.
- [ ] Change the output size to 5.

## ❓ Quick Questions

1. What does the hidden layer do?
2. What shape is the output for 5 samples?
3. What does `argmax(dim=1)` return?

---

## 🏁 Summary

> 💡 Layers + activation + loss + optimizer = a real (small) neural network.

⬅️ **Previous:** [Loss Functions](../04_loss_functions/loss_functions.md)
