# 🚀 MNIST Project: Model

> 📚 **Projects · Mnist Project** &nbsp;|&nbsp; 🧩 Lesson 1 of 3 &nbsp;|&nbsp; ⏱️ ~15 min &nbsp;|&nbsp; 🎓 🟣 Project
>
> Progress: `▰▱▱` 1/3

---

## 🎯 Learning Goal
Define the CNN used in the MNIST project.

## 📌 What is it?
The model file holds only the network definition so that `train.py` and `predict.py` can both import it. It has two convolution layers and two linear layers for 10 classes of handwritten digits.

## 🧠 Real-Life Analogy
Like a blueprint kept in one drawer so both the builders (train) and the inspectors (predict) use the same plans.

---

## 💻 Example

Code from `model.py` (run it with `python model.py`):

```python
# Model for the MNIST digits project
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)        # randomly switches off neurons while training
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))   # (32, 7, 7)
        x = x.flatten(start_dim=1)                # (1568)
        x = self.dropout(self.relu(self.fc1(x)))  # (64)
        return self.fc2(x)                        # (10)


if __name__ == "__main__":
    import torch
    model = Net()
    print(model)
    print("output shape:", tuple(model(torch.rand(2, 1, 28, 28)).shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
```

## 📤 Expected Output

```text
Net(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu): ReLU()
  (dropout): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=1568, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=10, bias=True)
)
output shape: (2, 10)
parameters: 105866

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. Two `Conv2d` layers (16 and 32 filters) with ReLU and pooling.
2. `Dropout(0.25)` randomly switches off neurons during training to reduce overfitting.
3. `fc1` and `fc2` turn the features into 10 class scores.
4. Running the file prints the model, an output shape and the parameter count.

---

## 📐 Shapes

```
(N, 1, 28, 28) -> conv1+pool -> (N, 16, 14, 14)
               -> conv2+pool -> (N, 32, 7, 7)
               -> flatten    -> (N, 1568)
               -> fc1+dropout-> (N, 64)
               -> fc2        -> (N, 10)
```

## 📝 Important Points

- ✅ Keeping the model in its own file avoids repeating code.
- ✅ Both train and predict must use the same class.
- ✅ The output size 10 = number of classes.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Changing the model after training | Saved weights must match the structure |
| Using dropout at test time | `model.eval()` turns it off |

## 🧪 Try It Yourself

- [ ] Change 64 to 128 hidden units.
- [ ] Run the file with a fake input of shape `(2, 1, 28, 28)`.
- [ ] Count the parameters.

## ❓ Quick Questions

1. Why keep the model in a separate file?
2. What is the output size and why?
3. What does pooling do?

---

## 🏁 Summary

> 💡 The model lives in its own file so training and prediction share it.

➡️ **Next:** [Train](../02_train/train.md)
