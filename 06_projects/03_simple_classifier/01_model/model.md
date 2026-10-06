# 🚀 Simple Classifier: Model

> 📚 **Projects · Simple Classifier** &nbsp;|&nbsp; 🧩 Lesson 1 of 3 &nbsp;|&nbsp; ⏱️ ~10 min &nbsp;|&nbsp; 🎓 🟣 Project
>
> Progress: `▰▱▱` 1/3

---

## 🎯 Learning Goal
Define a small classifier for 2D points.

## 📌 What is it?
This model takes two numbers (the x and y position of a point) and outputs two scores: 'outside the circle' or 'inside the circle'.

## 🧠 Real-Life Analogy
Like a bouncer checking one rule: is this person on the guest list (inside) or not (outside)?

---

## 💻 Example

Code from `model.py` (run it with `python model.py`):

```python
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
```

## 📤 Expected Output

```text
Classifier(
  (net): Sequential(
    (0): Linear(in_features=2, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=2, bias=True)
  )
)
output shape: (5, 2)
parameters: 354

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. `nn.Sequential` stacks Linear → ReLU → Linear → ReLU → Linear.
2. 2 inputs (x, y position) and 2 outputs (outside/inside).
3. Running the file prints the model, output shape and parameter count.

---

## 📐 Shapes

```
input  (N, 2)
-> (N, 16) -> (N, 16) -> (N, 2)
```

## 📝 Important Points

- ✅ A circle boundary is not a straight line, so we need hidden layers with ReLU.
- ✅ `Sequential` is perfect for simple stacks of layers.
- ✅ No dataset download is needed in this project.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Using one linear layer only | A circle needs hidden layers |
| Output size not 2 | One score per class |

## 🧪 Try It Yourself

- [ ] Change 16 to 32 hidden units.
- [ ] Remove one hidden layer and see if accuracy drops.
- [ ] Count the parameters.

## ❓ Quick Questions

1. Why can't a single linear layer solve this?
2. What do the 2 outputs mean?
3. What does `Sequential` do?

---

## 🏁 Summary

> 💡 A small Sequential network can learn a circle boundary.

➡️ **Next:** [Train](../02_train/train.md)
