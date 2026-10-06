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
