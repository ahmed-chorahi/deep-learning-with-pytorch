# 🏋️ Save Load Model

> 📚 **Training** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Save a trained model and load it again.

## 📌 What is it?
After training you don't want to start over. PyTorch lets you save the learned parameters (`state_dict`) to a file and load them into a fresh model later.

## 🧠 Real-Life Analogy
Like saving a video game. You can close the game and later load exactly where you were.

---

## 💻 Example

Code from `save_load_model.py` (run it with `python save_load_model.py`):

```python
# Lesson 5: Saving and loading a model

import os
import torch
import torch.nn as nn

torch.manual_seed(0)
here = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(here, "my_model.pth")
checkpoint_path = os.path.join(here, "checkpoint.pth")

print("=== 1. Save the weights (state_dict) ===")
model = nn.Linear(2, 1)
print("original weights:", model.weight.data)
torch.save(model.state_dict(), weights_path)
print("saved to my_model.pth")

print("\n=== 2. Load them into a new model ===")
new_model = nn.Linear(2, 1)                  # same structure first
print("new model (random):", new_model.weight.data)
new_model.load_state_dict(torch.load(weights_path))
new_model.eval()
print("after loading     :", new_model.weight.data)

x = torch.tensor([[1.0, 2.0]])
print("same prediction?", torch.equal(model(x), new_model(x)))

print("\n=== 3. A checkpoint also stores training progress ===")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
checkpoint = {
    "epoch": 5,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "loss": 0.123,
}
torch.save(checkpoint, checkpoint_path)

loaded = torch.load(checkpoint_path)
print("keys inside checkpoint:", list(loaded.keys()))
print("resume from epoch:", loaded["epoch"], "| saved loss:", loaded["loss"])

print("\n=== 4. What happens with the wrong structure? ===")
wrong = nn.Linear(3, 1)
try:
    wrong.load_state_dict(torch.load(weights_path))
except RuntimeError:
    print("Error: the saved weights do not fit a Linear(3, 1) model")

print("\n=== 5. Clean up the demo files ===")
os.remove(weights_path)
os.remove(checkpoint_path)
print("files removed")
```

## 📤 Expected Output

```text
=== 1. Save the weights (state_dict) ===
original weights: tensor([[-0.0053,  0.3793]])
saved to my_model.pth

=== 2. Load them into a new model ===
new model (random): tensor([[-0.5204, -0.2723]])
after loading     : tensor([[-0.0053,  0.3793]])
same prediction? True

=== 3. A checkpoint also stores training progress ===
keys inside checkpoint: ['epoch', 'model_state', 'optimizer_state', 'loss']
resume from epoch: 5 | saved loss: 0.123

=== 4. What happens with the wrong structure? ===
Error: the saved weights do not fit a Linear(3, 1) model

=== 5. Clean up the demo files ===
files removed
```

## 🔍 Explanation
1. Saves `state_dict()` to `my_model.pth`.
2. Loads it into a new model and proves the predictions match.
3. Saves a checkpoint (epoch, model, optimizer, loss) in a dictionary.
4. Loading into the wrong structure raises an error (caught).
5. Removes the demo files at the end.

---

## 📐 Save/Load Recipe

```python
torch.save(model.state_dict(), "model.pth")      # save

model = MyModel()                                # same structure first
model.load_state_dict(torch.load("model.pth"))   # load
model.eval()
```

## 📝 Important Points

- ✅ Save the `state_dict`, not the whole model object.
- ✅ The model class must exist before loading.
- ✅ Call `model.eval()` before predicting.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Loading into a different structure | Build the same model first |
| Forgetting `model.eval()` after loading | Call it before predicting |

## 🧪 Try It Yourself

- [ ] Save and load a bigger model, e.g. `SimpleNetwork`.
- [ ] Remove the `os.remove` line and look for the file.
- [ ] Try loading into a model of a different size and read the error.

## ❓ Quick Questions

1. What is a `state_dict`?
2. Why must the structure match when loading?
3. What should you call before predicting?

---

## 🏁 Summary

> 💡 Save the `state_dict`, rebuild the model, load the weights.

⬅️ **Previous:** [Validation](../04_validation/validation.md)
