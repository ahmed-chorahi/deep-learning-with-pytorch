# 🧠 Loss Functions

> 📚 **Neural Networks** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Learn how a loss function measures mistakes.

## 📌 What is it?
A **loss function** turns 'how wrong is the model?' into a single number. Training tries to make this number as small as possible.

## 🧠 Real-Life Analogy
A loss is like a test score for the model, but lower is better: 0 means a perfect answer.

---

## 💻 Example

Code from `loss_functions.py` (run it with `python loss_functions.py`):

```python
# Lesson 4: Loss functions
# A loss turns "how wrong is the model?" into one number. Lower is better.

import torch
import torch.nn as nn

print("=== 1. Regression losses (predicting numbers) ===")
prediction = torch.tensor([2.5, 0.0, 2.0])
target = torch.tensor([3.0, -0.5, 2.0])

mse = nn.MSELoss()(prediction, target)
mae = nn.L1Loss()(prediction, target)
print("MSE:", round(mse.item(), 4))
print("MAE:", round(mae.item(), 4))

# Same answers by hand
print("MSE by hand:", ((prediction - target) ** 2).mean().item())
print("MAE by hand:", (prediction - target).abs().mean().item())

print("\n=== 2. Classification loss (CrossEntropyLoss) ===")
logits = torch.tensor([[2.0, 0.5, 0.1],     # sample 1 -> model likes class 0
                       [0.1, 0.2, 3.0]])    # sample 2 -> model likes class 2
good_labels = torch.tensor([0, 2])
bad_labels = torch.tensor([2, 0])

ce = nn.CrossEntropyLoss()
print("loss with correct labels:", round(ce(logits, good_labels).item(), 4))
print("loss with wrong labels  :", round(ce(logits, bad_labels).item(), 4))

print("\n=== 3. Cross entropy step by step ===")
probs = torch.softmax(logits, dim=1)
print("probabilities:\n", probs.round(decimals=3))
picked = probs[range(2), good_labels]       # probability of the true class
print("probability of true class:", picked.round(decimals=3))
print("loss by hand:", (-torch.log(picked)).mean().item())

print("\n=== 4. Binary classification (yes / no) ===")
yes_no_logits = torch.tensor([2.0, -1.0, 0.5])
truth = torch.tensor([1.0, 0.0, 1.0])
bce = nn.BCEWithLogitsLoss()
print("BCE loss:", round(bce(yes_no_logits, truth).item(), 4))

print("\n=== 5. Loss gets bigger as the guess gets worse ===")
true_value = torch.tensor([10.0])
for guess in [10.0, 9.0, 7.0, 3.0]:
    loss = nn.MSELoss()(torch.tensor([guess]), true_value)
    print(f"guess {guess:5.1f} -> loss {loss.item():6.2f}  {'#' * int(loss.item() // 2)}")
```

## 📤 Expected Output

```text
=== 1. Regression losses (predicting numbers) ===
MSE: 0.1667
MAE: 0.3333
MSE by hand: 0.1666666716337204
MAE by hand: 0.3333333432674408

=== 2. Classification loss (CrossEntropyLoss) ===
loss with correct labels: 0.2132
loss with wrong labels  : 2.6132

=== 3. Cross entropy step by step ===
probabilities:
 tensor([[0.7280, 0.1630, 0.1090],
        [0.0490, 0.0540, 0.8960]])
probability of true class: tensor([0.7280, 0.8960])
loss by hand: 0.21319007873535156

=== 4. Binary classification (yes / no) ===
BCE loss: 0.3048

=== 5. Loss gets bigger as the guess gets worse ===
guess  10.0 -> loss   0.00  
guess   9.0 -> loss   1.00  
guess   7.0 -> loss   9.00  ####
guess   3.0 -> loss  49.00  ########################
```

## 🔍 Explanation
1. MSE and MAE for regression, checked by hand.
2. `CrossEntropyLoss` with correct vs wrong labels: wrong labels give a bigger loss.
3. Cross entropy step by step: softmax, pick the true class probability, `-log`.
4. `BCEWithLogitsLoss` for yes/no problems.
5. A small text chart shows loss growing as the guess gets worse.

---

## 📐 Which Loss to Use

| Task | Loss |
|------|------|
| Predict a number | `MSELoss` |
| Choose 1 of N classes | `CrossEntropyLoss` |
| Yes / no | `BCELoss` (or `BCEWithLogitsLoss`) |

`CrossEntropyLoss` input: logits `(batch, classes)`, labels `(batch,)`.

## 📝 Important Points

- ✅ Lower loss = better predictions.
- ✅ Give `CrossEntropyLoss` raw scores, not probabilities.
- ✅ The loss must be a single number before `backward()`.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Giving probabilities to `CrossEntropyLoss` | Give raw scores (logits) |
| Using float labels for classification | Labels must be integer class numbers |

## 🧪 Try It Yourself

- [ ] Compute MSE by hand for `[1, 2]` vs `[2, 4]` (answer 2.5).
- [ ] Make logits that strongly favor the wrong class and compare the loss.
- [ ] Try `L1Loss` on the same data.

## ❓ Quick Questions

1. What does a loss function measure?
2. Which loss is used for classification?
3. What does `CrossEntropyLoss` expect as input?

---

## 🏁 Summary

> 💡 A loss is the model's score for being wrong; training tries to minimize it.

⬅️ **Previous:** [Activation Functions](../03_activation_functions/activation_functions.md)  |  ➡️ **Next:** [Simple Network](../05_simple_network/simple_network.md)
