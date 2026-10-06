# 🖼️ Building a CNN

> 📚 **CNN (Convolutional Neural Networks)** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
Build a clean CNN class with a feature extractor and a classifier.

## 📌 What is it?
A typical CNN has two parts: a **feature extractor** (`Conv → ReLU → Pool`, repeated) and a **classifier** (`Flatten → Linear`). The class works out the flatten size automatically using a dummy image.

## 🧠 Real-Life Analogy
Like a quality-control line: first stations look for details (edges, shapes), then the last station decides what the item is.

---

## 💻 Example

Code from `building_cnn.py` (run it with `python building_cnn.py`):

```python
# Lesson 3: Building a CNN
# Pattern: [Conv -> ReLU -> Pool] repeated, then Flatten -> Linear.

import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, num_classes=3, image_size=12):
        super().__init__()
        # feature extractor: finds patterns in the image
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # work out how many numbers come out of the feature extractor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            flat_size = self.features(dummy).flatten(start_dim=1).shape[1]
        # classifier: makes the final decision
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = MyCNN()
print("=== 1. The model ===")
print(model)

print("\n=== 2. Shape after every layer ===")
x = torch.rand(4, 1, 12, 12)                  # 4 images, 1 channel, 12x12
print(f"{'input':28s} {tuple(x.shape)}")
for layer in list(model.features) + list(model.classifier):
    x = layer(x)
    print(f"{layer.__class__.__name__:12s} {'':15s} {tuple(x.shape)}")

print("\n=== 3. Parameters per layer ===")
total = 0
for name, p in model.named_parameters():
    total += p.numel()
    print(f"{name:22s} {str(tuple(p.shape)):16s} {p.numel():5d}")
print("total:", total)

print("\n=== 4. Different image size ===")
bigger = MyCNN(num_classes=10, image_size=28)
print("28x28 model output:", tuple(bigger(torch.rand(2, 1, 28, 28)).shape))
print("28x28 model parameters:", sum(p.numel() for p in bigger.parameters()))
```

## 📤 Expected Output

```text
=== 1. The model ===
MyCNN(
  (features): Sequential(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=144, out_features=32, bias=True)
    (2): ReLU()
    (3): Linear(in_features=32, out_features=3, bias=True)
  )
)

=== 2. Shape after every layer ===
input                        (4, 1, 12, 12)
Conv2d                       (4, 8, 12, 12)
ReLU                         (4, 8, 12, 12)
MaxPool2d                    (4, 8, 6, 6)
Conv2d                       (4, 16, 6, 6)
ReLU                         (4, 16, 6, 6)
MaxPool2d                    (4, 16, 3, 3)
Flatten                      (4, 144)
Linear                       (4, 32)
ReLU                         (4, 32)
Linear                       (4, 3)

=== 3. Parameters per layer ===
features.0.weight      (8, 1, 3, 3)        72
features.0.bias        (8,)                 8
features.3.weight      (16, 8, 3, 3)     1152
features.3.bias        (16,)               16
classifier.1.weight    (32, 144)         4608
classifier.1.bias      (32,)               32
classifier.3.weight    (3, 32)             96
classifier.3.bias      (3,)                 3
total: 5987

=== 4. Different image size ===
28x28 model output: (2, 10)
28x28 model parameters: 26698

Note: random numbers are used, so your values will differ.
```

## 🔍 Explanation
1. `self.features` holds the convolution and pooling layers.
2. A dummy tensor is passed through to count the numbers going into the linear layer.
3. `self.classifier` flattens and applies two linear layers.
4. Part 2 prints the shape after every layer.
5. Part 3 lists parameters per layer and the total.
6. Part 4 makes a version for 28×28 images with 10 classes.

---

## 📐 Shapes (12×12 input)

```
input          (N, 1, 12, 12)
Conv  + ReLU   (N, 8, 12, 12)
MaxPool        (N, 8, 6, 6)
Conv  + ReLU   (N, 16, 6, 6)
MaxPool        (N, 16, 3, 3)
Flatten        (N, 144)
Linear + ReLU  (N, 32)
Linear         (N, 3)
```

## 📝 Important Points

- ✅ `nn.Sequential` keeps simple stacks tidy.
- ✅ Most parameters are usually in the first linear layer.
- ✅ Number of output classes = size of the last layer.
- ✅ Use a dummy input to avoid counting sizes by hand.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Hard-coding the flatten size | Compute it with a dummy tensor |
| Forgetting `padding` | The image shrinks quickly |

## 🧪 Try It Yourself

- [ ] Add a third convolution layer.
- [ ] Change filters from 8/16 to 4/8.
- [ ] Build the model for 32×32 colour images (3 channels).

## ❓ Quick Questions

1. What are the two parts of a CNN?
2. Why use a dummy input?
3. Where are most parameters?

---

## 🏁 Summary

> 💡 A CNN = feature extractor (Conv/ReLU/Pool) + classifier (Flatten/Linear).

⬅️ **Previous:** [Pooling Padding Stride](../02_pooling_padding_stride/pooling_padding_stride.md)  |  ➡️ **Next:** [Cnn Training](../04_cnn_training/cnn_training.md)
