# 🧪 Final Practice

> 📚 **Practice** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Put everything together in one small project.

## 📌 What is it?
You build 3 groups of random 2D points, split them into train/test, train a small network to tell the groups apart, evaluate it, and predict a new point.

## 🧠 Real-Life Analogy
Like the final exam project: use every skill from the whole course in one go.

---

## 💻 Example

Code from `final_practice.py` (run it with `python final_practice.py`):

```python
# Final practice: put everything together
# Goal: train a small network to classify points into 3 groups.
# Try changing the numbers marked "TRY" and see what happens!
import torch
import torch.nn as nn

torch.manual_seed(1)

# --- 1. Make fake data: 3 groups of points around different centers ---
centers = torch.tensor([[0.0, 0.0], [3.0, 3.0], [0.0, 4.0]])
x = torch.cat([center + torch.randn(100, 2) for center in centers])
y = torch.cat([torch.full((100,), i) for i in range(3)])

# --- 2. Shuffle and split into train / test ---
order = torch.randperm(300)
x, y = x[order], y[order]
x_train, y_train = x[:240], y[:240]
x_test, y_test = x[240:], y[240:]
print("train:", tuple(x_train.shape), "| test:", tuple(x_test.shape))

# --- 3. Build the model ---
HIDDEN = 16                                   # TRY: 4, 32
model = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 3))

# --- 4. Loss and optimizer ---
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)    # TRY: 0.1, 0.001

# --- 5. Train ---
for epoch in range(100):                      # TRY: 20, 300
    model.train()
    loss = loss_fn(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | loss = {loss.item():.4f}")

# --- 6. Evaluate ---
model.eval()
with torch.no_grad():
    predictions = model(x_test).argmax(dim=1)
accuracy = (predictions == y_test).float().mean().item()
print(f"Test accuracy: {accuracy * 100:.1f}%")

# --- 7. Confusion table (rows = true group, columns = predicted group) ---
print("\nConfusion table:")
print("        pred0 pred1 pred2")
for true_group in range(3):
    row = [((predictions == p) & (y_test == true_group)).sum().item() for p in range(3)]
    print(f"true {true_group}  {row[0]:5d} {row[1]:5d} {row[2]:5d}")

# --- 8. Predict a new point ---
new_point = torch.tensor([[2.8, 3.2]])
with torch.no_grad():
    probs = torch.softmax(model(new_point), dim=1)[0]
print("\nNew point [2.8, 3.2] -> group", probs.argmax().item(), "(expected 1)")
print("probabilities:", [round(p, 2) for p in probs.tolist()])
```

## 📤 Expected Output

```text
train: (240, 2) | test: (60, 2)
Epoch   0 | loss = 1.4640
Epoch  20 | loss = 0.3969
Epoch  40 | loss = 0.2174
Epoch  60 | loss = 0.1750
Epoch  80 | loss = 0.1609
Test accuracy: 93.3%

Confusion table:
        pred0 pred1 pred2
true 0     18     0     0
true 1      0    16     2
true 2      2     0    22

New point [2.8, 3.2] -> group 1 (expected 1)
probabilities: [0.0, 0.99, 0.01]
```

## 🔍 Explanation
1. Creates 3 clusters of 100 points each.
2. Shuffles and splits into 240 train / 60 test points.
3. Builds `Linear → ReLU → Linear`.
4. Trains with Adam for 100 epochs.
5. Evaluates accuracy and prints a confusion table.
6. Predicts a new point.

---

## 📐 Data Shapes

```
x       (300, 2)   300 points, 2 coordinates
y       (300,)     group 0, 1 or 2
x_train (240, 2)   x_test (60, 2)
model   (N, 2) -> (N, 16) -> (N, 3)
```

## 📝 Important Points

- ✅ This uses the whole pipeline: data, model, loss, optimizer, loop, evaluation.
- ✅ Lines marked `TRY` are meant to be changed.
- ✅ Results are repeatable thanks to the seed.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Not shuffling before the split | Shuffle with `randperm` |

## 🧪 Try It Yourself

- [ ] Change `HIDDEN` to 4, then 32.
- [ ] Change the learning rate to 0.1.
- [ ] Add a 4th group of points.

## ❓ Quick Questions

1. What are the main steps of this project?
2. Why split train and test?
3. Which number is the output size and why?

---

## 🏁 Summary

> 💡 The whole workflow in one small project.

⬅️ **Previous:** [Pytorch Interview](../04_pytorch_interview/pytorch_interview.md)
