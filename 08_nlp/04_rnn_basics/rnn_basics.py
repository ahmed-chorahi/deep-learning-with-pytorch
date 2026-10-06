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
