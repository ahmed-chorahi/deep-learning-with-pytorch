# Lesson 4: Predicting one image
# Train a small CNN for a short time, then ask it about a single picture.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# Same model as in 03_cnn/cnn.py (copied so this lesson runs on its own)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.flatten(start_dim=1))


torch.manual_seed(0)
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

small_train = Subset(train_data, range(5000))      # only 5000 images -> quick
loader = DataLoader(small_train, batch_size=64, shuffle=True)

model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("=== 1. Training for 2 epochs ===")
for epoch in range(2):
    for images, labels in loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} done, last loss = {loss.item():.3f}")

print("\n=== 2. Predicting one image ===")
model.eval()
image, true_label = test_data[0]
with torch.no_grad():
    logits = model(image.unsqueeze(0))              # add the batch dimension
    probs = torch.softmax(logits, dim=1)[0]
predicted = probs.argmax().item()
print("True label:", true_label)
print("Predicted :", predicted)
print("Confidence:", round(probs[predicted].item() * 100, 1), "%")

print("\n=== 3. Top 3 guesses ===")
top_probs, top_digits = probs.topk(3)
for p, d in zip(top_probs, top_digits):
    print(f"digit {d.item()}: {p.item() * 100:5.1f}% {'#' * int(p.item() * 30)}")

print("\n=== 4. Predicting 10 images ===")
correct = 0
for i in range(10):
    img, label = test_data[i]
    with torch.no_grad():
        guess = model(img.unsqueeze(0)).argmax(dim=1).item()
    mark = "ok" if guess == label else "WRONG"
    correct += guess == label
    print(f"image {i}: true = {label}, predicted = {guess}  {mark}")
print(f"{correct}/10 correct")
