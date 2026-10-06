# 🧪 Neural Network Questions

> 📚 **Practice** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Review layers, losses and the training step.

## 📌 What is it?
Six questions on building networks with `nn.Sequential`, counting parameters, choosing a loss and running one training step.

## 🧠 Real-Life Analogy
Like a pop quiz that checks whether the basics have stuck.

---

## 💻 Example

Code from `neural_network_questions.py` (run it with `python neural_network_questions.py`):

```python
# Neural network questions
import torch
import torch.nn as nn

# Q1: Build a network with 10 inputs, one hidden layer of 20 neurons (ReLU), 3 outputs
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
print("Q1:", model)

# Q2: How many parameters does nn.Linear(10, 20) have?  (10*20 weights + 20 biases)
count = sum(p.numel() for p in nn.Linear(10, 20).parameters())
print("Q2:", count, "= 10*20 + 20")

# Q3: What shape comes out for a batch of 8 samples?
print("Q3:", tuple(model(torch.rand(8, 10)).shape))

# Q4: Which loss for classification into 3 classes?  CrossEntropyLoss
logits = model(torch.rand(8, 10))
labels = torch.randint(0, 3, (8,))
print("Q4: loss =", round(nn.CrossEntropyLoss()(logits, labels).item(), 4))

# Q5: One training step
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss()(model(torch.rand(8, 10)), labels)   # forward + loss
optimizer.zero_grad()                                           # clear gradients
loss.backward()                                                 # backpropagate
optimizer.step()                                                # update weights
print("Q5: one training step done")

# Q6: Why do we need activation functions?
print("Q6: Without them, stacked linear layers collapse into one linear layer.")

# Q7: Count the parameters of the whole Q1 network
total = sum(p.numel() for p in model.parameters())
print("Q7:", total, "= 220 + 63")

# Q8: Train the network on a tiny made-up problem and watch the loss fall
torch.manual_seed(0)
x = torch.rand(30, 10)
y = (x.sum(dim=1) > 5).long()                  # class 1 if the numbers add up to > 5
net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.Adam(net.parameters(), lr=0.05)
for epoch in range(51):
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 25 == 0:
        print(f"Q8: epoch {epoch:2d} loss = {loss.item():.3f}")
```

## 📤 Expected Output

```text
Q1: Sequential(
  (0): Linear(in_features=10, out_features=20, bias=True)
  (1): ReLU()
  (2): Linear(in_features=20, out_features=3, bias=True)
)
Q2: 220 = 10*20 + 20
Q3: (8, 3)
Q4: loss = 1.1522
Q5: one training step done
Q6: Without them, stacked linear layers collapse into one linear layer.
Q7: 283 = 220 + 63
Q8: epoch  0 loss = 0.702
Q8: epoch 25 loss = 0.063
Q8: epoch 50 loss = 0.007
```

## 🔍 Explanation
1. Q1: build a 10→20→3 network.
2. Q2: count parameters (220).
3. Q3: output shape for a batch.
4. Q4: use `CrossEntropyLoss`.
5. Q5: one training step.
6. Q6: why activations matter.
7. Q7: total parameters (283).
8. Q8: train on a tiny made-up problem and watch the loss.

---

## 📐 Parameter Count

```
Linear(in, out) parameters = in * out + out
Linear(10, 20) = 10*20 + 20 = 220
```

## 📝 Important Points

- ✅ Parameters = weights + biases.
- ✅ Output size = number of classes.
- ✅ One training step = zero_grad, backward, step.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting biases when counting | Parameters = in × out + out |

## 🧪 Try It Yourself

- [ ] Count the parameters of the whole Q1 network (220 + 63 = 283).
- [ ] Change Q1 to 5 outputs.
- [ ] Write Q5 as a loop of 10 steps.

## ❓ Quick Questions

1. How many parameters in `Linear(10, 20)`?
2. Which loss for classification?
3. Why activation functions?

---

## 🏁 Summary

> 💡 Eight questions about layers, losses and training steps.

⬅️ **Previous:** [Tensor Questions](../01_tensor_questions/tensor_questions.md)  |  ➡️ **Next:** [Cnn Questions](../03_cnn_questions/cnn_questions.md)
