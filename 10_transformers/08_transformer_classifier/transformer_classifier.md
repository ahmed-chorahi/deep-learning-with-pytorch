# 🤖 Text Classification with a Transformer

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 8 of 10 &nbsp;|&nbsp; ⏱️ ~35 min &nbsp;|&nbsp; 🎓 🟠 Upper-Intermediate
>
> Progress: `▰▰▰▰▰▰▰▰▱▱` 8/10

---

## 🎯 Learning Goal
Train a Transformer encoder to classify sentences as positive or negative.

## 📌 What is it?
We put the encoder to work: after the blocks we average the vectors of the real words and use a Linear layer to get two scores (negative / positive). Training is the same loop you already know.

## 🧠 Real-Life Analogy
A panel of experts (the blocks) discusses the sentence, then one spokesperson (the linear layer) announces the verdict.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Masked mean** | average only over real words, ignoring `<pad>` |
| **Classifier head** | the final Linear layer that gives class scores |
| **TensorDataset** | wraps tensors so DataLoader can batch them |

---

## 💻 Example

Code from `transformer_classifier.py` (run it with `python transformer_classifier.py`):

```python
# Lesson 8: Text classification with a Transformer encoder
# Pipeline: ids -> Embedding + positions -> Transformer blocks -> average -> Linear

import math
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

# ----- vocabulary and encoding (right padding this time: the mask hides the pads) -----
words = sorted({w for s, _ in train_data for w in s.split()})
word_to_id = {"<pad>": 0, "<unk>": 1}
for w in words:
    word_to_id[w] = len(word_to_id)
MAX_LEN = 7


def encode(sentence):
    ids = [word_to_id.get(w, 1) for w in sentence.split()][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


def to_dataset(data):
    return TensorDataset(torch.tensor([encode(s) for s, _ in data]),
                         torch.tensor([label for _, label in data]))


train_loader = DataLoader(to_dataset(train_data), batch_size=8, shuffle=True)
test_x, test_y = to_dataset(test_data).tensors


def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ffn(x)))


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=32, num_heads=4, d_ff=64, num_layers=2, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("pe", sinusoidal_encoding(MAX_LEN, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        pad_mask = ids == 0                                           # (batch, words)
        x = self.embedding(ids) * math.sqrt(self.d_model) + self.pe[:ids.shape[1]]
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        # average only over the REAL words (ignore padding)
        real = (~pad_mask).unsqueeze(-1).float()                      # (batch, words, 1)
        sentence_vector = (x * real).sum(dim=1) / real.sum(dim=1)
        return self.classifier(sentence_vector)


model = TransformerClassifier(len(word_to_id))
print("vocabulary size:", len(word_to_id))
print("parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


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
parameters: 18242

=== Training ===
epoch 10 | loss = 0.0078 | test accuracy = 100%
epoch 20 | loss = 0.0014 | test accuracy = 100%
epoch 30 | loss = 0.0010 | test accuracy = 100%
epoch 40 | loss = 0.0008 | test accuracy = 100%
epoch 50 | loss = 0.0006 | test accuracy = 100%
epoch 60 | loss = 0.0005 | test accuracy = 100%

=== Predicting new sentences ===
'what a great film'          -> positive (100%)
'i hate this awful movie'    -> negative (100%)
'so happy and nice'          -> positive (100%)
'boring and sad'             -> negative (100%)
```

## 🔍 Explanation
1. 24 training and 4 test sentences are encoded with a vocabulary; `<pad>` = 0 and `<unk>` = 1.
2. Sentences are padded on the **right** this time, because the padding mask hides the extra tokens.
3. `TransformerClassifier` combines embedding, positions, blocks and a classifier.
4. The forward pass computes a masked average: `(x * real).sum(dim=1) / real.sum(dim=1)`.
5. The usual five-step training loop runs for 60 epochs with Adam.
6. Test accuracy is printed every 10 epochs.
7. Four new sentences are predicted with confidence.

---

## 📐 Model Flow

```
sentence ──► ids (batch, 7)
              │ Embedding + positions
              ▼
          (batch, 7, 32)
              │ 2 × TransformerBlock (padding mask)
              ▼
          (batch, 7, 32)
              │ masked mean over real words
              ▼
          (batch, 32)
              │ Linear(32, 2)
              ▼
          (batch, 2)   ──► [negative score, positive score]
```

Compare with lesson 5 of the NLP folder: there an LSTM read the sentence in order; here attention reads it all at once.

## 📝 Important Points

- ✅ Transformers use the same training loop as every other model.
- ✅ Use the masked mean so padding does not dilute the average.
- ✅ A tiny dataset is easy; real tasks need much more data.
- ✅ Always check accuracy on sentences not used for training.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Averaging over padding too | Use the mask to average real words only |
| Forgetting `model.eval()` when testing | Dropout would still be active |

## 🧪 Try It Yourself

- [ ] Add 4 new training sentences.
- [ ] Change `num_layers` to 1 and compare.
- [ ] Try a sentence with only unknown words.

## ❓ Quick Questions

1. Why do we use a masked mean?
2. What are the main parts of `TransformerClassifier`?
3. How is this different from the LSTM classifier?

---

## 🏁 Summary

> 💡 Encoder + masked average + Linear is a complete Transformer text classifier.

⬅️ **Previous:** [Transformer Encoder](../07_transformer_encoder/transformer_encoder.md)  |  ➡️ **Next:** [Causal Language Model](../09_causal_language_model/causal_language_model.md)
