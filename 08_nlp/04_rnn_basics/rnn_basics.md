# 💬 RNN Basics: Reading Text in Order

> 📚 **NLP (Natural Language Processing)** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Learn how RNN, LSTM and GRU layers process sequences and what their shapes are.

## 📌 What is it?
A **Recurrent Neural Network (RNN)** reads a sentence one word at a time and carries a hidden state (its memory) from word to word. **LSTM** and **GRU** are improved versions that remember things for longer.

## 🧠 Real-Life Analogy
Reading a book: you don't start from zero on every page; you carry a summary of the story so far in your head. The hidden state is that summary.

---

## 💻 Example

Code from `rnn_basics.py` (run it with `python rnn_basics.py`):

```python
# Lesson 4: RNN basics
# Normal layers ignore word order. An RNN reads a sentence one word at a time
# and keeps a "memory" (the hidden state) of what it has seen so far.

import torch
import torch.nn as nn

torch.manual_seed(0)

batch_size, seq_len, input_size, hidden_size = 2, 5, 4, 3
x = torch.rand(batch_size, seq_len, input_size)     # 2 sentences, 5 words, 4 numbers per word
print("input shape:", tuple(x.shape), "= (batch, words, vector size)")

print("\n=== 1. nn.RNN ===")
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
output, hidden = rnn(x)
print("output shape:", tuple(output.shape), "= hidden state at EVERY word")
print("hidden shape:", tuple(hidden.shape), "= final hidden state (layers, batch, hidden)")
print("last output equals hidden?", torch.allclose(output[:, -1, :], hidden[0]))

print("\n=== 2. The same thing, one word at a time (RNNCell) ===")
cell = nn.RNNCell(input_size, hidden_size)
h = torch.zeros(batch_size, hidden_size)            # memory starts empty
for t in range(seq_len):
    h = cell(x[:, t, :], h)
    print(f"after word {t + 1}: memory of sentence 1 = {h[0].detach().round(decimals=2).tolist()}")

print("\n=== 3. LSTM and GRU (RNNs with better memory) ===")
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
output, (hidden, cell_state) = lstm(x)              # LSTM also returns a cell state
print("LSTM output:", tuple(output.shape), "| hidden:", tuple(hidden.shape), "| cell:", tuple(cell_state.shape))
gru = nn.GRU(input_size, hidden_size, batch_first=True)
output, hidden = gru(x)
print("GRU  output:", tuple(output.shape), "| hidden:", tuple(hidden.shape))

print("\n=== 4. Parameter counts ===")
for name, layer in [("RNN", rnn), ("GRU", gru), ("LSTM", lstm)]:
    print(f"{name:5s}: {sum(p.numel() for p in layer.parameters())}")

print("\n=== 5. Two layers and both directions ===")
deep = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
output, (hidden, _) = deep(x)
print("output:", tuple(output.shape), "= hidden size doubled (forward + backward)")
print("hidden:", tuple(hidden.shape), "= (layers * directions, batch, hidden)")

print("\n=== 6. Using the last hidden state to classify ===")
classifier = nn.Linear(hidden_size, 2)
_, (hidden, _) = lstm(x)
scores = classifier(hidden[-1])
print("class scores shape:", tuple(scores.shape), "= (batch, classes)")
```

## 📤 Expected Output

```text
input shape: (2, 5, 4) = (batch, words, vector size)

=== 1. nn.RNN ===
output shape: (2, 5, 3) = hidden state at EVERY word
hidden shape: (1, 2, 3) = final hidden state (layers, batch, hidden)
last output equals hidden? True

=== 2. The same thing, one word at a time (RNNCell) ===
after word 1: memory of sentence 1 = [0.6899999976158142, 0.23999999463558197, 0.6200000047683716]
after word 2: memory of sentence 1 = [0.5299999713897705, 0.7900000214576721, 0.20000000298023224]
after word 3: memory of sentence 1 = [0.28999999165534973, 0.2800000011920929, 0.3799999952316284]
after word 4: memory of sentence 1 = [0.41999998688697815, 0.4699999988079071, 0.18000000715255737]
after word 5: memory of sentence 1 = [0.5, 0.25, 0.5899999737739563]

=== 3. LSTM and GRU (RNNs with better memory) ===
LSTM output: (2, 5, 3) | hidden: (1, 2, 3) | cell: (1, 2, 3)
GRU  output: (2, 5, 3) | hidden: (1, 2, 3)

=== 4. Parameter counts ===
RNN  : 27
GRU  : 81
LSTM : 108

=== 5. Two layers and both directions ===
output: (2, 5, 6) = hidden size doubled (forward + backward)
hidden: (4, 2, 3) = (layers * directions, batch, hidden)

=== 6. Using the last hidden state to classify ===
class scores shape: (2, 2) = (batch, classes)
```

## 🔍 Explanation
1. The input has shape `(2, 5, 4)`: 2 sentences, 5 words, 4 numbers per word.
2. `nn.RNN` returns `output` (hidden state at every word) and `hidden` (the final one).
3. `RNNCell` shows the same idea in a loop, so you can watch the memory change.
4. `LSTM` also returns a cell state; `GRU` is similar but simpler.
5. Parameter counts: LSTM has the most, then GRU, then RNN.
6. `num_layers=2` and `bidirectional=True` change the output shapes.
7. The last hidden state goes through `nn.Linear` to make class scores.

---

## 📐 Shapes in an RNN

```
x       : (batch, words, input_size)        (2, 5, 4)
output  : (batch, words, hidden_size)       (2, 5, 3)   memory at every word
hidden  : (layers, batch, hidden_size)      (1, 2, 3)   final memory
```

```
word 1 ──► [RNN] ──► h1 ──► [RNN] ──► h2 ──► [RNN] ──► h3 ...
                  (memory is passed along)
```

| Layer | Idea | Parameters (this example) |
|-------|------|---------------------------|
| RNN | simple memory | 27 |
| GRU | gates decide what to keep | 81 |
| LSTM | gates + separate cell memory | 108 |

## 📝 Important Points

- ✅ Use `batch_first=True` so the batch comes first.
- ✅ `output[:, -1]` is the same as the final hidden state for a one-layer RNN.
- ✅ LSTM returns `(hidden, cell)`; RNN and GRU return only `hidden`.
- ✅ Bidirectional layers double the output size.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting `batch_first=True` | PyTorch then expects `(words, batch, size)` |
| Using `hidden` as if it had a batch first shape | `hidden` is `(layers, batch, size)` |

## 🧪 Try It Yourself

- [ ] Change `seq_len` to 8 and check the shapes.
- [ ] Make `hidden_size` 6 and count the parameters again.
- [ ] Print `output[:, 0, :]` and compare with the first memory step.

## ❓ Quick Questions

1. What is a hidden state?
2. What is the difference between `output` and `hidden`?
3. Why do LSTMs exist?

---

## 🏁 Summary

> 💡 RNNs read one word at a time and pass a memory along; LSTM/GRU remember longer.

⬅️ **Previous:** [Bag Of Words](../03_bag_of_words/bag_of_words.md)  |  ➡️ **Next:** [Text Classifier](../05_text_classifier/text_classifier.md)
