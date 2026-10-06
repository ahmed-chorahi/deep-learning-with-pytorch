# 💬 Text Classifier with an LSTM

> 📚 **NLP (Natural Language Processing)** &nbsp;|&nbsp; 🧩 Lesson 5 of 5 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▰` 5/5

---

## 🎯 Learning Goal
Put everything together: tokenization, embedding, LSTM and a classifier.

## 📌 What is it?
This lesson builds a complete sentiment model: `Embedding → LSTM → Linear`. It is trained on 24 short sentences and tested on 4 unseen ones.

## 🧠 Real-Life Analogy
Like a reader who learns each word's meaning (embedding), reads the sentence while remembering (LSTM), and finally gives a verdict (linear layer).

---

## 💻 Example

Code from `text_classifier.py` (run it with `python text_classifier.py`):

```python
# Lesson 5: Text classifier with an LSTM
# Pipeline: words -> ids -> Embedding -> LSTM -> Linear -> positive / negative

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

train_data = [
    ("i love this movie", 1), ("what a great film", 1), ("good acting and nice story", 1),
    ("i am so happy", 1), ("great fun and good music", 1), ("nice and lovely", 1),
    ("love the happy ending", 1), ("a good great movie", 1), ("so nice i love it", 1),
    ("wonderful and happy", 1), ("this is a wonderful film", 1), ("happy and fun", 1),
    ("i hate this movie", 0), ("what an awful film", 0), ("bad acting and boring story", 0),
    ("i am so sad", 0), ("awful and bad music", 0), ("boring and ugly", 0),
    ("hate the sad ending", 0), ("a bad awful movie", 0), ("so boring i hate it", 0),
    ("terrible and sad", 0), ("this is a terrible film", 0), ("sad and boring", 0),
]
test_data = [
    ("what a lovely happy film", 1), ("great music and nice acting", 1),
    ("a sad and boring movie", 0), ("i hate the awful story", 0),
]

# ----- vocabulary and encoding -----
words = sorted({w for s, _ in train_data for w in s.split()})
word_to_id = {"<pad>": 0, "<unk>": 1}
for w in words:
    word_to_id[w] = len(word_to_id)
MAX_LEN = 7


def encode(sentence):
    ids = [word_to_id.get(w, 1) for w in sentence.split()][:MAX_LEN]
    # pad on the LEFT so the last word is always a real word
    return [0] * (MAX_LEN - len(ids)) + ids


def to_dataset(data):
    x = torch.tensor([encode(s) for s, _ in data])
    y = torch.tensor([label for _, label in data])
    return TensorDataset(x, y)


train_loader = DataLoader(to_dataset(train_data), batch_size=8, shuffle=True)
test_x, test_y = to_dataset(test_data).tensors


# ----- model -----
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)              # (batch, words, embed_dim)
        _, (hidden, _) = self.lstm(x)      # hidden: (1, batch, hidden_dim)
        return self.fc(hidden[-1])         # (batch, 2)


model = TextClassifier(len(word_to_id))
print("vocabulary size:", len(word_to_id))
print("parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def test_accuracy():
    model.eval()
    with torch.no_grad():
        return (model(test_x).argmax(dim=1) == test_y).float().mean().item() * 100


print("\n=== Training ===")
for epoch in range(1, 61):
    model.train()
    total = 0.0
    for x, y in train_loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    if epoch % 10 == 0:
        print(f"epoch {epoch:2d} | loss = {total / len(train_loader):.4f} | test accuracy = {test_accuracy():.0f}%")

print("\n=== Predicting new sentences ===")
model.eval()
for sentence in ["what a great film", "i hate this awful movie", "so happy and nice", "boring and sad"]:
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor([encode(sentence)])), dim=1)[0]
    label = "positive" if probs.argmax().item() == 1 else "negative"
    print(f"{sentence!r:28s} -> {label:8s} ({probs.max().item() * 100:.0f}%)")
```

## 📤 Expected Output

```text
vocabulary size: 34
parameters: 2754

=== Training ===
epoch 10 | loss = 0.0707 | test accuracy = 100%
epoch 20 | loss = 0.0012 | test accuracy = 100%
epoch 30 | loss = 0.0007 | test accuracy = 100%
epoch 40 | loss = 0.0005 | test accuracy = 100%
epoch 50 | loss = 0.0004 | test accuracy = 100%
epoch 60 | loss = 0.0003 | test accuracy = 100%

=== Predicting new sentences ===
'what a great film'          -> positive (100%)
'i hate this awful movie'    -> negative (100%)
'so happy and nice'          -> positive (100%)
'boring and sad'             -> negative (100%)
```

## 🔍 Explanation
1. A vocabulary is built from the training sentences with `<pad>` and `<unk>`.
2. `encode()` converts a sentence to ids and **left-pads** it to 7 words, so the last word is always real.
3. `TensorDataset` and `DataLoader` make shuffled batches.
4. `TextClassifier` has an embedding, an LSTM and a linear layer; `forward` uses the last hidden state.
5. The training loop prints loss and test accuracy every 10 epochs.
6. At the end the model predicts four new sentences with confidence.

---

## 📐 Model Flow

```
sentence ─► ids (7) ─► Embedding ─► LSTM ─► last hidden ─► Linear ─► [neg, pos]
 (batch,7)   (batch,7,16)       (batch,7,16)    (batch,16)       (batch,2)
```

## 📝 Important Points

- ✅ Left padding keeps the last position real, which matters because we use the last hidden state.
- ✅ `padding_idx=0` keeps `<pad>` at zero.
- ✅ Use `model.eval()` and `no_grad` for testing.
- ✅ More data is needed for real-world text.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Right-padding and using the last hidden state | The last step would be padding |
| Testing on training sentences | Keep a separate test set |

## 🧪 Try It Yourself

- [ ] Add 4 more training sentences.
- [ ] Change `hidden_dim` to 8.
- [ ] Try a sentence with only unknown words.

## ❓ Quick Questions

1. Why do we pad on the left here?
2. What does `hidden[-1]` give us?
3. What are the three layers of the model?

---

## 🏁 Summary

> 💡 Embedding + LSTM + Linear is the standard first text-classification model.

⬅️ **Previous:** [Rnn Basics](../04_rnn_basics/rnn_basics.md)
