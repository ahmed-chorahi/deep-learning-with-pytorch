# 🚀 Simple Classifier: Predict

> 📚 **Projects · Simple Classifier** &nbsp;|&nbsp; 🧩 Lesson 3 of 3 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟣 Project
>
> Progress: `▰▰▰` 3/3

---

## 🎯 Learning Goal
Predict with the trained circle classifier.

## 📌 What is it?
This script loads the saved model and classifies four new points as inside or outside the circle.

## 🧠 Real-Life Analogy
Like asking a trained guard about four visitors: inside the fence or outside?

---

## 💻 Example

Code from `predict.py` (run it with `python predict.py`):

```python
# Load the trained classifier and predict new points (run ../02_train/train.py first)
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_model"))
from model import Classifier

model = Classifier()
model.load_state_dict(torch.load(os.path.join(HERE, "..", "02_train", "model.pth")))
model.eval()

# ----- 1. Predict a few points we choose -----
new_points = torch.tensor([[0.0, 0.0],      # centre -> inside
                           [0.5, 0.5],      # inside
                           [1.5, 1.5],      # outside
                           [-1.8, 0.2]])    # outside

with torch.no_grad():
    probs = torch.softmax(model(new_points), dim=1)
predictions = probs.argmax(dim=1)

for point, pred, p in zip(new_points, predictions, probs):
    answer = "inside" if pred.item() == 1 else "outside"
    x, y = point.tolist()
    print(f"Point ({x:5.1f}, {y:5.1f}) -> {answer:7s} (confidence {p[pred].item() * 100:.0f}%)")

# ----- 2. Draw what the model learned -----
print("\nModel's map (# = inside, . = outside):")
for y in [2 - 0.25 * i for i in range(17)]:             # top to bottom
    row = ""
    for x in [-2 + 0.125 * j for j in range(33)]:       # left to right
        with torch.no_grad():
            guess = model(torch.tensor([[x, y]])).argmax(dim=1).item()
        row += "#" if guess == 1 else "."
    print(row)
```

## 📤 Expected Output

```text
Point (  0.0,   0.0) -> inside  (confidence 100%)
Point (  0.5,   0.5) -> inside  (confidence 100%)
Point (  1.5,   1.5) -> outside (confidence 100%)
Point ( -1.8,   0.2) -> outside (confidence 100%)

Model's map (# = inside, . = outside):
.................................
.................................
.................................
.................................
.................................
...........###########...........
..........#############..........
.........###############.........
.........################........
.........###############.........
.........##############..........
...........###########...........
.................................
.................................
.................................
.................................
.................................
```

## 🔍 Explanation
1. Loads the trained model.
2. Predicts four chosen points with confidence.
3. Draws a text map of the whole area to show what the model learned.

---

## 📐 Points Used

| Point | Expected |
|-------|----------|
| (0, 0) | inside |
| (0.5, 0.5) | inside |
| (1.5, 1.5) | outside |
| (-1.8, 0.2) | outside |

## 📝 Important Points

- ✅ Run `02_train/train.py` first.
- ✅ Points close to the edge are the hardest.
- ✅ Printed floats like -1.7999999 are normal rounding of float32.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Running without training | Run `../02_train/train.py` first |

## 🧪 Try It Yourself

- [ ] Add the point (0.9, 0.0) and (1.1, 0.0).
- [ ] Print the probabilities with softmax.
- [ ] Test 100 random points.

## ❓ Quick Questions

1. What does class 1 mean here?
2. Why might points near the edge be wrong?
3. What must be run before this script?

---

## 🏁 Summary

> 💡 The text map shows the circle the model has learned.

⬅️ **Previous:** [Train](../02_train/train.md)
