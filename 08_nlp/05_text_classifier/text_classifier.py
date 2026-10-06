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
