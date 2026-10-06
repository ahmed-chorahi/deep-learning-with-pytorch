# 🤖 A Tiny GPT: Generating Text

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 9 of 10 &nbsp;|&nbsp; ⏱️ ~40 min &nbsp;|&nbsp; 🎓 🟠 Upper-Intermediate
>
> Progress: `▰▰▰▰▰▰▰▰▰▱` 9/10

---

## 🎯 Learning Goal
Build a small decoder-style Transformer that predicts the next character and generates text.

## 📌 What is it?
A **language model** predicts the next token. GPT-style models use a **causal mask** so every position can only look at earlier positions. We train on a short text, one character at a time, then generate new text by repeatedly picking the next character.

## 🧠 Real-Life Analogy
Autocomplete on your phone: it looks at what you typed so far and guesses the next letter. It must not peek at letters you haven't typed yet.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Causal mask** | hides future positions with `-inf` |
| **Next-token prediction** | target = the input shifted by one |
| **Logits** | a score for every character in the vocabulary |
| **Greedy decoding** | always pick the character with the highest score |
| **Block size** | how many characters the model sees at once |

---

## 💻 Example

Code from `causal_language_model.py` (run it with `python causal_language_model.py`):

```python
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
```

## 📤 Expected Output

```text
characters in the text: 558 | vocabulary size: 20

=== 1. The causal mask ===
tensor([[0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]], dtype=torch.int32)
row i can only look at columns 0..i (the past and itself)

=== 2. Input and target are shifted by one character ===
input : formers like attention. 
target: ormers like attention. a

=== 3. The model ===
parameters: 104084
output shape for a batch: (2, 24, 20) = (batch, characters, vocab)
loss of random guessing = ln(vocab) = 3.0

=== 4. Training ===
step   1 | loss = 3.1580
step 100 | loss = 0.1142
step 200 | loss = 0.0923
step 300 | loss = 0.0875
step 400 | loss = 0.0923

=== 5. Generating text ===
'i like '        -> i like pytorch. i like transformers. transformers like attention. a
'attention '     -> attention is all you need. i like pytorch. i like transformers. transf
'transformers '  -> transformers like attention. attention is all you need. i like pytorch. i

=== 6. Sampling gives variety (greedy always picks the top choice) ===
next character 'p': 50%
next character 't': 44%
next character 'a': 4%
```

## 🔍 Explanation
1. Part 1 prints a causal mask: `True` marks positions that may **not** be seen.
2. `get_batch` cuts random windows; the target is the same window shifted by one character.
3. Part 2 prints an input and its target to show the shift.
4. `TinyGPT` has token embeddings, learned position embeddings, two blocks with the causal mask and a Linear head to the vocabulary.
5. Part 3 prints the parameter count (104,084) and the output shape `(batch, characters, vocab)`.
6. Part 4 trains for 400 steps; the loss falls from about 3.2 (random guessing) to about 0.1.
7. `generate` feeds the last 24 characters into the model, takes the best next character and repeats.
8. Part 5 completes three prompts; Part 6 shows the top next-character probabilities.

---

## 📐 Next-Character Prediction

```
text:    i  l  i  k  e  _  p  y
input :  i  l  i  k  e  _  p       (positions 0..6)
target:  _  l  i  k  e  _  p  y    (shifted left by one)
```

**Causal mask for 5 positions** (`1` = not allowed to look):

```
0 1 1 1 1
0 0 1 1 1
0 0 0 1 1
0 0 0 0 1
0 0 0 0 0
```

| Part | Shape |
|------|-------|
| input ids | `(batch, 24)` |
| after embeddings | `(batch, 24, 64)` |
| logits | `(batch, 24, vocab_size)` |

The loss compares the logits at *every* position with the next character.

**Why does it just repeat the text?** The model is tiny and trained on a few sentences, so it memorizes them. Real models like GPT learn the same way but on enormous amounts of text.

## 📝 Important Points

- ✅ The causal mask is what makes a model a *generator*.
- ✅ Targets are the inputs shifted by one position.
- ✅ Reshape logits to `(batch*seq, vocab)` before `CrossEntropyLoss`.
- ✅ Generation is a loop: predict, append, repeat.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| No causal mask | The model would simply copy the next character and learn nothing |
| Giving the model a context longer than `BLOCK` | Only pass the last `BLOCK` characters |

## 🧪 Try It Yourself

- [ ] Change the text to your own sentences.
- [ ] Train for 100 steps instead of 400 and see the generated text.
- [ ] Replace greedy choice with `torch.multinomial` sampling.

## ❓ Quick Questions

1. What does the causal mask do?
2. Why is the target shifted by one character?
3. How does the model generate text?

---

## 🏁 Summary

> 💡 Causal mask + next-token prediction + a generation loop = a tiny GPT.

⬅️ **Previous:** [Transformer Classifier](../08_transformer_classifier/transformer_classifier.md)  |  ➡️ **Next:** [Transformer Practice](../10_transformer_practice/transformer_practice.md)
