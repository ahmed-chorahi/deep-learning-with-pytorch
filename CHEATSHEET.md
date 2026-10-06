# 📝 PyTorch Cheat Sheet

A one-page reminder of everything in this repository. Keep it open while you code!

---

## 🔢 Tensors

| I want to... | Code |
|--------------|------|
| make a tensor | `torch.tensor([1, 2, 3])` |
| zeros / ones / random | `torch.zeros(2, 3)` · `torch.ones(2, 3)` · `torch.rand(2, 3)` |
| a range of numbers | `torch.arange(0, 10, 2)` |
| check the shape | `x.shape` |
| change the shape | `x.reshape(3, 4)` · `x.reshape(2, -1)` |
| add / remove a dimension | `x.unsqueeze(0)` · `x.squeeze()` |
| reorder dimensions | `x.permute(2, 0, 1)` |
| matrix multiplication | `a @ b` |
| sum / mean over a dimension | `x.sum(dim=0)` · `x.mean(dim=1)` |
| index of the biggest value | `x.argmax()` |
| filter with a condition | `x[x > 5]` |
| a real copy | `x.clone()` |
| use the GPU if available | `device = "cuda" if torch.cuda.is_available() else "cpu"` |

## 📈 Autograd

```python
x = torch.tensor(3.0, requires_grad=True)   # track this tensor
y = x ** 2
y.backward()                                # compute gradients
print(x.grad)                               # dy/dx = 6

with torch.no_grad():                       # no tracking (predictions)
    ...
```

## 🧠 Building a model

```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

| Task | Output layer | Loss |
|------|--------------|------|
| Predict a number | `Linear(n, 1)` | `nn.MSELoss()` |
| Pick 1 of N classes | `Linear(n, N)` | `nn.CrossEntropyLoss()` (raw scores, no softmax) |
| Yes / no | `Linear(n, 1)` | `nn.BCEWithLogitsLoss()` |

## 🏋️ The training loop (memorize this!)

```python
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        prediction = model(x)            # 1. forward
        loss = loss_fn(prediction, y)    # 2. loss
        optimizer.zero_grad()            # 3. clear old gradients
        loss.backward()                  # 4. backpropagation
        optimizer.step()                 # 5. update weights

    model.eval()
    with torch.no_grad():                # validation / testing
        ...
```

## 💾 Saving and loading

```python
torch.save(model.state_dict(), "model.pth")

model = MyNet()                                   # same structure first
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

## 👁️ Images and CNNs

| Item | Remember |
|------|----------|
| Image shape | `(batch, channels, height, width)` |
| Convolution size | `(size - kernel + 2*padding) / stride + 1` |
| `padding=1` with a 3×3 kernel | keeps the size |
| `MaxPool2d(2)` | halves height and width |
| Flatten size | channels × height × width |
| Predict one image | `model(image.unsqueeze(0)).argmax(dim=1)` |

## 💬 Text (NLP)

| Step | Code |
|------|------|
| words → ids | vocabulary dictionary with `<pad>` = 0 and `<unk>` = 1 |
| ids → vectors | `nn.Embedding(vocab_size, dim)` |
| read in order | `nn.LSTM(dim, hidden, batch_first=True)` |
| shapes | embedding output `(batch, words, dim)` |

## 🤖 Transformers

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

| Part | Job |
|------|-----|
| Positional encoding | tells the model the word order |
| Multi-head attention | words look at other words (several views) |
| Feed-forward | processes each word separately |
| Residual + LayerNorm | keeps training stable |
| Causal mask | stops a word from seeing the future (GPT) |
| Padding mask | ignores `<pad>` tokens |

## 🐞 Common errors and quick fixes

| Error message | Usual cause | Fix |
|---------------|-------------|-----|
| `shape '[...]' is invalid for input` | wrong `reshape` | check the number of elements |
| `mat1 and mat2 shapes cannot be multiplied` | wrong `Linear` size | print `x.shape` before the layer |
| `Expected input batch_size to match target batch_size` | shapes of output and labels differ | check `y.shape` |
| `element 0 of tensors does not require grad` | tracking was off | no `no_grad` during training |
| loss is `nan` | learning rate too high | try a smaller `lr` |
| model never improves | forgot `zero_grad()` or wrong loss | re-check the 5 steps |
