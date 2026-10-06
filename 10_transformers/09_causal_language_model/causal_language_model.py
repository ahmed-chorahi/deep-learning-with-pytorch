# Lesson 9: A tiny GPT-style language model
# Task: given the characters so far, predict the NEXT character.
# A causal mask stops each position from looking at the future.
# We generate text one character at a time.

import math
import torch
import torch.nn as nn

torch.manual_seed(0)

text = ("i like pytorch. i like transformers. transformers like attention. "
        "attention is all you need. ") * 6
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
data = torch.tensor([char_to_id[c] for c in text])
vocab_size = len(chars)
BLOCK = 24                                              # how many characters the model sees at once
print("characters in the text:", len(text), "| vocabulary size:", vocab_size)

print("\n=== 1. The causal mask ===")
mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()  # True = NOT allowed to look
print(mask.int())
print("row i can only look at columns 0..i (the past and itself)")


def get_batch(batch_size=32):
    starts = torch.randint(0, len(data) - BLOCK - 1, (batch_size,))
    x = torch.stack([data[s:s + BLOCK] for s in starts])             # input characters
    y = torch.stack([data[s + 1:s + BLOCK + 1] for s in starts])     # the same text shifted by one
    return x, y


x, y = get_batch(2)
print("\n=== 2. Input and target are shifted by one character ===")
print("input :", "".join(id_to_char[i] for i in x[0].tolist()))
print("target:", "".join(id_to_char[i] for i in y[0].tolist()))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(BLOCK, d_model)           # learned positions
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)                       # a score for every next character

    def forward(self, ids):
        seq = ids.shape[1]
        positions = torch.arange(seq)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        causal = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, causal)
        return self.head(x)                                              # (batch, seq, vocab)


model = TinyGPT(vocab_size)
print("\n=== 3. The model ===")
print("parameters:", sum(p.numel() for p in model.parameters()))
print("output shape for a batch:", tuple(model(x).shape), "= (batch, characters, vocab)")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
print("loss of random guessing = ln(vocab) =", round(math.log(vocab_size), 2))

print("\n=== 4. Training ===")
for step in range(1, 401):
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))        # flatten batch and time
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0 or step == 1:
        print(f"step {step:3d} | loss = {loss.item():.4f}")


def generate(prompt, length=60):
    model.eval()
    ids = [char_to_id[c] for c in prompt]
    for _ in range(length):
        context = torch.tensor([ids[-BLOCK:]])                           # last BLOCK characters
        with torch.no_grad():
            logits = model(context)[0, -1]                               # scores for the next character
        ids.append(logits.argmax().item())                               # pick the most likely one
    return "".join(id_to_char[i] for i in ids)


print("\n=== 5. Generating text ===")
for prompt in ["i like ", "attention ", "transformers "]:
    print(f"{prompt!r:16s} ->", generate(prompt))

print("\n=== 6. Sampling gives variety (greedy always picks the top choice) ===")
model.eval()
context = torch.tensor([[char_to_id[c] for c in "i like "]])
with torch.no_grad():
    probs = torch.softmax(model(context)[0, -1], dim=0)
top = probs.topk(3)
for p, i in zip(top.values, top.indices):
    print(f"next character {id_to_char[i.item()]!r}: {p.item() * 100:.0f}%")
