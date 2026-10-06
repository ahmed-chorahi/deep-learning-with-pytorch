# 🖼️ Training a CNN on Our Own Images

> 📚 **CNN (Convolutional Neural Networks)** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Train a CNN from start to finish on images we generate ourselves.

## 📌 What is it?
We draw tiny 12×12 images with a line (horizontal, vertical or diagonal) plus noise, and teach a CNN to recognize the direction. Nothing needs to be downloaded.

## 🧠 Real-Life Analogy
Like teaching a child to tell the direction of a scribble after showing hundreds of noisy examples.

---

## 💻 Example

Code from `cnn_training.py` (run it with `python cnn_training.py`):

```python
# Lesson 4: Training a CNN
# We make our own tiny images so nothing needs to be downloaded.
# Class 0 = horizontal line, class 1 = vertical line, class 2 = diagonal line.

import torch
import torch.nn as nn

torch.manual_seed(0)
SIZE = 12
names = ["horizontal", "vertical", "diagonal"]


def make_image(label):
    img = torch.zeros(1, SIZE, SIZE)
    pos = torch.randint(2, SIZE - 2, (1,)).item()
    if label == 0:
        img[0, pos, :] = 1.0              # horizontal line at a random row
    elif label == 1:
        img[0, :, pos] = 1.0              # vertical line at a random column
    else:
        for i in range(SIZE):
            img[0, i, i] = 1.0            # diagonal line
    img += 0.3 * torch.rand_like(img)     # add some noise
    return img


def make_dataset(n):
    labels = torch.randint(0, 3, (n,))
    images = torch.stack([make_image(l.item()) for l in labels])
    return images, labels


x_train, y_train = make_dataset(600)
x_test, y_test = make_dataset(150)
print("train images:", tuple(x_train.shape), "| test images:", tuple(x_test.shape))

print("\n=== An example image (class: %s) ===" % names[y_train[0]])
for row in x_train[0, 0]:
    print("".join("#" if p > 0.6 else "." for p in row))

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # (8, 6, 6)
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # (16, 3, 3)
    nn.Flatten(),
    nn.Linear(16 * 3 * 3, 3),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def accuracy(x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item() * 100


print("\n=== Training ===")
print(f"before training: test accuracy = {accuracy(x_test, y_test):.1f}%")
batch_size = 50
for epoch in range(1, 16):
    model.train()
    order = torch.randperm(len(x_train))             # shuffle every epoch
    total = 0.0
    for start in range(0, len(x_train), batch_size):
        idx = order[start:start + batch_size]
        loss = loss_fn(model(x_train[idx]), y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    if epoch % 3 == 0:
        print(f"epoch {epoch:2d} | loss = {total / 12:.4f} | test accuracy = {accuracy(x_test, y_test):.1f}%")

print("\n=== Accuracy for each class ===")
model.eval()
with torch.no_grad():
    predictions = model(x_test).argmax(dim=1)
for c in range(3):
    mask = y_test == c
    acc = (predictions[mask] == c).float().mean().item() * 100
    print(f"{names[c]:11s} {acc:5.1f}% {'#' * int(acc // 5)}")

print("\n=== Predict one new image ===")
new_image = make_image(2).unsqueeze(0)               # add the batch dimension
with torch.no_grad():
    probs = torch.softmax(model(new_image), dim=1)[0]
print("probabilities:", {n: round(p, 2) for n, p in zip(names, probs.tolist())})
print("prediction:", names[probs.argmax().item()], "(true: diagonal)")
```

## 📤 Expected Output

```text
train images: (600, 1, 12, 12) | test images: (150, 1, 12, 12)

=== An example image (class: diagonal) ===
#...........
.#..........
..#.........
...#........
....#.......
.....#......
......#.....
.......#....
........#...
.........#..
..........#.
...........#

=== Training ===
before training: test accuracy = 64.0%
epoch  3 | loss = 0.0004 | test accuracy = 100.0%
epoch  6 | loss = 0.0000 | test accuracy = 100.0%
epoch  9 | loss = 0.0000 | test accuracy = 100.0%
epoch 12 | loss = 0.0000 | test accuracy = 100.0%
epoch 15 | loss = 0.0000 | test accuracy = 100.0%

=== Accuracy for each class ===
horizontal  100.0% ####################
vertical    100.0% ####################
diagonal    100.0% ####################

=== Predict one new image ===
probabilities: {'horizontal': 0.0, 'vertical': 0.0, 'diagonal': 1.0}
prediction: diagonal (true: diagonal)
```

## 🔍 Explanation
1. `make_image(label)` draws a line at a random position and adds noise.
2. 600 training and 150 test images are created.
3. One example image is printed as text.
4. The CNN uses two Conv/ReLU/Pool blocks and one linear layer.
5. The training loop shuffles with `randperm` and uses batches of 50.
6. Accuracy is printed for each class.
7. A new image is predicted with probabilities.

---

## 📐 The Three Classes

| Class | Name | What the image shows |
|-------|------|----------------------|
| 0 | horizontal | a line across one row |
| 1 | vertical | a line down one column |
| 2 | diagonal | a line from top-left to bottom-right |

Data shapes: images `(600, 1, 12, 12)`, labels `(600,)`.

## 📝 Important Points

- ✅ Making your own data is great for experiments.
- ✅ Shuffle the training data every epoch.
- ✅ Accuracy before training is about chance (about 33% for 3 classes).
- ✅ This task is easy, so a high accuracy is expected.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Using test images to choose settings | Keep them for the end |
| Forgetting `unsqueeze(0)` for one image | A model expects a batch |

## 🧪 Try It Yourself

- [ ] Add a 4th class (anti-diagonal).
- [ ] Add more noise (`0.6`) and see the accuracy.
- [ ] Train for 3 epochs only.

## ❓ Quick Questions

1. Why do we add noise to the images?
2. What is chance accuracy for 3 classes?
3. Why do we shuffle every epoch?

---

## 🏁 Summary

> 💡 Make data, build a CNN, train, test per class and predict a new image.

⬅️ **Previous:** [Building Cnn](../03_building_cnn/building_cnn.md)  |  ➡️ **Next:** [Batchnorm Dropout](../05_batchnorm_dropout/batchnorm_dropout.md)
