# Task: is a 2D point inside a circle of radius 1 (class 1) or outside (class 0)?
# We create our own data, so nothing needs to be downloaded.
import os
import sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
# The model class lives in the sibling folder 01_model
sys.path.insert(0, os.path.join(HERE, "..", "01_model"))
from model import Classifier

torch.manual_seed(42)


def make_data(n):
    points = torch.rand(n, 2) * 4 - 2                       # random points in [-2, 2]
    labels = ((points ** 2).sum(dim=1) < 1).long()          # inside the circle?
    return points, labels


# ----- data: train / validation / test -----
x_train, y_train = make_data(2000)
x_val, y_val = make_data(500)
x_test, y_test = make_data(500)
print("points inside the circle in training data:", y_train.sum().item(), "of", len(y_train))


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item() * 100


# ----- model, loss, optimizer -----
model = Classifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----- training -----
best_val = 0.0
for epoch in range(1, 201):
    model.train()
    loss = loss_fn(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_acc = accuracy(model, x_val, y_val)
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(HERE, "model.pth"))

    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | loss = {loss.item():.4f} | validation accuracy = {val_acc:.1f}%")

# ----- final test with the best saved model -----
model.load_state_dict(torch.load(os.path.join(HERE, "model.pth")))
print(f"Test accuracy: {accuracy(model, x_test, y_test):.1f}%")
print("Best model saved to model.pth")
