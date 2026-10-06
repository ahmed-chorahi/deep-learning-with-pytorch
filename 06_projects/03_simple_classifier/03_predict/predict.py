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
