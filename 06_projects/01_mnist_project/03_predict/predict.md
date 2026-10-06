# 🚀 MNIST Project: Predict

> 📚 **Projects · Mnist Project** &nbsp;|&nbsp; 🧩 Lesson 3 of 3 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟣 Project
>
> Progress: `▰▰▰` 3/3

---

## 🎯 Learning Goal
Load the trained MNIST model and make predictions.

## 📌 What is it?
This script builds the model, loads `model.pth`, and predicts the first 5 test images, printing the true label, the predicted label and the confidence.

## 🧠 Real-Life Analogy
Like using a trained assistant: you hand over new examples and they tell you the answer and how sure they are.

---

## 💻 Example

Code from `predict.py` (run it with `python predict.py`):

```python
# Load the trained model and predict test images (run ../02_train/train.py first)
import os
import sys
import torch
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_model"))
from model import Net

CLASS_NAMES = [str(i) for i in range(10)]

# ----- load the model -----
model = Net()
model.load_state_dict(torch.load(os.path.join(HERE, "..", "02_train", "model.pth")))
model.eval()

test_data = datasets.MNIST(root=os.path.join(HERE, "..", "data"), train=False,
                         download=True, transform=transforms.ToTensor())

# ----- predict the first 8 test images -----
correct = 0
for i in range(8):
    image, label = test_data[i]
    with torch.no_grad():
        probs = torch.softmax(model(image.unsqueeze(0)), dim=1)[0]
    predicted = probs.argmax().item()
    correct += predicted == label
    print(f"Image {i}: true = {CLASS_NAMES[label]:12s} | predicted = {CLASS_NAMES[predicted]:12s} "
          f"| confidence = {probs[predicted].item() * 100:5.1f}%")
print(f"{correct}/8 correct")

# ----- draw one image with text -----
image, label = test_data[0]
print("\nFirst test image (true label: %s):" % CLASS_NAMES[label])
for row in image[0][::2]:
    print("".join("#" if p > 0.5 else "." for p in row))
```

## 📤 Expected Output

```text
Image 0: true = 7            | predicted = 7            | confidence =  99.x%
Image 1: true = 2            | predicted = 2            | confidence =  99.x%
... (8 images in total)
8/8 correct   (or 7/8, depending on training)

First test image: (a text picture made of # and .)
```

## 🔍 Explanation
1. Loads the saved weights into a new `Net`.
2. Predicts the first 8 test images with confidence.
3. Counts how many are correct.
4. Draws the first test image with text.

---

## 📐 Prediction Flow

```
image (1, 28, 28) -> unsqueeze(0) -> (1, 1, 28, 28)
model -> scores (1, 10) -> softmax -> probabilities
argmax -> predicted class
```

## 📝 Important Points

- ✅ Run `02_train/train.py` first, or loading will fail.
- ✅ The model structure must match the saved weights.
- ✅ High confidence does not guarantee the answer is right.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Missing `model.pth` | Run `../02_train/train.py` first |
| Forgetting `eval()` | Dropout would still be active |

## 🧪 Try It Yourself

- [ ] Predict 10 images instead of 5.
- [ ] Print only the wrong predictions.
- [ ] Predict a single image of your choice.

## ❓ Quick Questions

1. Why is `model.eval()` called?
2. What file must exist before running predict?
3. How is confidence calculated?

---

## 🏁 Summary

> 💡 Load the trained MNIST model and use it.

⬅️ **Previous:** [Train](../02_train/train.md)
