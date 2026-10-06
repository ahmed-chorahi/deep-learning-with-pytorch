# Train the model on MNIST digits and save the best version
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
# The model class lives in the sibling folder 01_model
sys.path.insert(0, os.path.join(HERE, "..", "01_model"))
from model import Net

# ----- settings -----
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# ----- data -----
transform = transforms.ToTensor()
data_folder = os.path.join(HERE, "..", "data")
train_data = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=data_folder, train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)
print("train images:", len(train_data), "| test images:", len(test_data))

# ----- model, loss, optimizer -----
model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch():
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test_accuracy():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return 100 * correct / len(test_data)


# ----- training -----
best_accuracy = 0.0
for epoch in range(1, EPOCHS + 1):
    avg_loss = train_one_epoch()
    accuracy = test_accuracy()
    print(f"Epoch {epoch}/{EPOCHS} | loss = {avg_loss:.4f} | test accuracy = {accuracy:.2f}%")
    if accuracy > best_accuracy:                 # keep only the best model
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(HERE, "model.pth"))

print(f"Best accuracy: {best_accuracy:.2f}% (model saved to model.pth)")
