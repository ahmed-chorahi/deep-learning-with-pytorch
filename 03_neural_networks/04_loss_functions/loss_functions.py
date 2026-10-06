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
