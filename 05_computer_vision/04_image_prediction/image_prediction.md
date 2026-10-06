# 👁️ Image Prediction

> 📚 **Computer Vision** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Train briefly and predict a single image.

## 📌 What is it?
This lesson trains the CNN on 5,000 MNIST images for 2 epochs and then predicts one test image, showing the predicted digit and the model's confidence.

## 🧠 Real-Life Analogy
Like a student who studied a little and is now asked to read one handwritten digit out loud, saying how sure they are.

---

## 💻 Example

Code from `image_prediction.py` (run it with `python image_prediction.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. Training for 2 epochs ===
Epoch 1 done, last loss = 0.4xx
Epoch 2 done, last loss = 0.2xx

=== 2. Predicting one image ===
True label: 7
Predicted : 7
Confidence: 9x.x %

=== 3. Top 3 guesses ===
digit 7:  9x.x% ############################
digit 2:   x.x% 
digit 9:   x.x% 

=== 4. Predicting 10 images ===
image 0: true = 7, predicted = 7  ok
image 1: true = 2, predicted = 2  ok
...
9/10 correct

(Numbers vary a little each run. Needs internet for the first MNIST download.)
```

## 🔍 Explanation
1. Trains a small CNN for 2 epochs on 5,000 MNIST images.
2. Predicts one test image with `unsqueeze(0)`, `softmax` and `argmax`.
3. Prints the top 3 guesses with a text bar.
4. Predicts 10 images and counts how many are correct.

---

## 📐 Predicting One Image

```
image            (1, 28, 28)
unsqueeze(0)  -> (1, 1, 28, 28)   add batch
model(...)    -> (1, 10)          class scores
softmax       -> probabilities that sum to 1
argmax        -> the predicted digit
```

## 📝 Important Points

- ✅ A model always expects a batch, even for one image.
- ✅ Confidence is the highest softmax probability.
- ✅ More training gives better predictions.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting `model.eval()` | Call it before predicting |
| No batch dimension | Use `unsqueeze(0)` |

## 🧪 Try It Yourself

- [ ] Predict image number 5.
- [ ] Train for 5 epochs and compare.
- [ ] Print the top 3 predicted digits.

## ❓ Quick Questions

1. Why use `unsqueeze(0)`?
2. What does softmax output?
3. What is `argmax` used for?

---

## 🏁 Summary

> 💡 To predict: eval mode, add a batch, softmax, argmax.

⬅️ **Previous:** [Cnn](../03_cnn/cnn.md)  |  ➡️ **Next:** [Test Images](../05_test_images/test_images.md)
