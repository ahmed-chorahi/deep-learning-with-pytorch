# 🤖 Positional Encoding: Teaching Order

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 5 of 10 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▰▰▱▱▱▱▱` 5/10

---

## 🎯 Learning Goal
Understand why Transformers need position information and how sinusoidal and learned encodings work.

## 📌 What is it?
Self-attention treats a sentence like a bag of words, so 'dog bites man' and 'man bites dog' look the same. A **positional encoding** is a vector for each position that is *added* to the word vector, so the model knows where each word is.

## 🧠 Real-Life Analogy
Imagine seat numbers in a cinema. Two people with the same name are different to the usher because they sit in different seats.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Position** | the index of a word: 0, 1, 2... |
| **Sinusoidal encoding** | a fixed formula using sine and cosine waves |
| **Learned positions** | a trainable `nn.Embedding` with one row per position |
| **max_len** | the longest sentence the table supports |

---

## 💻 Example

Code from `positional_encoding.py` (run it with `python positional_encoding.py`):

```python
# Lesson 5: Positional encoding
# Attention ignores word order. So we ADD a position signal to every word vector.

import math
import torch
import torch.nn as nn

torch.manual_seed(0)


def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()               # (max_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)      # even columns: sine
    pe[:, 1::2] = torch.cos(position * div)      # odd columns: cosine
    return pe


pe = sinusoidal_encoding(max_len=20, d_model=8)

print("=== 1. The encoding table ===")
print("shape:", tuple(pe.shape), "= (positions, d_model)")
print("position 0:", pe[0].round(decimals=2).tolist())
print("position 1:", pe[1].round(decimals=2).tolist())
print("position 2:", pe[2].round(decimals=2).tolist())

print("\n=== 2. A text picture of the table (rows = position, columns = dimension) ===")
shades = " .:-=+*#%@"
big = sinusoidal_encoding(16, 16)
for pos in range(16):
    row = ""
    for value in big[pos]:
        row += shades[int((value.item() + 1) / 2 * 9)] * 2
    print(f"pos {pos:2d} |{row}|")

print("\n=== 3. How similar are different positions? ===")
# (with only 8 dimensions the pattern is rough; real models use hundreds)
for other in [1, 2, 5, 10, 19]:
    sim = torch.cosine_similarity(pe[0], pe[other], dim=0).item()
    print(f"similarity(position 0, position {other:2d}) = {sim:5.2f}")

print("\n=== 4. Adding position to word vectors ===")
embedding = nn.Embedding(10, 8)
ids = torch.tensor([[3, 4, 3]])                      # the SAME word (3) appears twice
words = embedding(ids)
with_pos = words + pe[:3]
print("same word, no position   -> first and last vectors equal?", torch.equal(words[0, 0], words[0, 2]))
print("same word, with position -> first and last vectors equal?", torch.equal(with_pos[0, 0], with_pos[0, 2]))

print("\n=== 5. Why order matters ===")
# 'dog bites man' and 'man bites dog' use the same words but mean different things.
# A tiny self-attention (no learned weights) shows what the model "sees" for the word 'bites'.


def attend(x):
    weights = torch.softmax(x @ x.transpose(-2, -1) / math.sqrt(x.shape[-1]), dim=-1)
    return weights @ x


a = embedding(torch.tensor([[1, 2, 3]]))             # dog bites man
b = embedding(torch.tensor([[3, 2, 1]]))             # man bites dog
bites_a = attend(a)[0, 1]                            # result for 'bites' (position 1)
bites_b = attend(b)[0, 1]
print("without position, 'bites' looks the same in both sentences?", torch.allclose(bites_a, bites_b, atol=1e-6))

bites_a_pos = attend(a + pe[:3])[0, 1]
bites_b_pos = attend(b + pe[:3])[0, 1]
print("with position, 'bites' looks the same in both sentences?   ", torch.allclose(bites_a_pos, bites_b_pos, atol=1e-6))

print("\n=== 6. Learned positions (another common choice) ===")
pos_embedding = nn.Embedding(20, 8)                  # a trainable table, one row per position
positions = torch.arange(3).unsqueeze(0)             # [[0, 1, 2]]
print("learned position vectors shape:", tuple(pos_embedding(positions).shape))
print("trainable parameters:", pos_embedding.weight.numel())
print("sinusoidal parameters: 0 (it is a fixed formula)")
```

## 📤 Expected Output

```text
=== 1. The encoding table ===
shape: (20, 8) = (positions, d_model)
position 0: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
position 1: [0.8399999737739563, 0.5400000214576721, 0.10000000149011612, 1.0, 0.009999999776482582, 1.0, 0.0, 1.0]
position 2: [0.9100000262260437, -0.41999998688697815, 0.20000000298023224, 0.9800000190734863, 0.019999999552965164, 1.0, 0.0, 1.0]

=== 2. A text picture of the table (rows = position, columns = dimension) ===
pos  0 |==@@==@@==@@==@@==@@==@@==@@==@@|
pos  1 |%%**++%%==%%==%%==%%==%%==%%==%%|
pos  2 |%%::##%%++%%==%%==%%==%%==%%==%%|
pos  3 |++  %%##++%%==%%==%%==%%==%%==%%|
pos  4 |....%%++**%%++%%==%%==%%==%%==%%|
pos  5 |  ++%%==**%%++%%==%%==%%==%%==%%|
pos  6 |--%%%%--##%%++%%==%%==%%==%%==%%|
pos  7 |####%%..####++%%==%%==%%==%%==%%|
pos  8 |%%--##  ####++%%==%%==%%==%%==%%|
pos  9 |**  ++  %%##++%%==%%==%%==%%==%%|
pos 10 |::  ==  %%**++%%==%%==%%==%%==%%|
pos 11 |  ==--  %%****%%==%%==%%==%%==%%|
pos 12 |::%%..  %%****%%++%%==%%==%%==%%|
pos 13 |**%%  ..%%++**%%++%%==%%==%%==%%|
pos 14 |%%++  --%%++**%%++%%==%%==%%==%%|
pos 15 |##..  ==%%==**%%++%%==%%==%%==%%|

=== 3. How similar are different positions? ===
similarity(position 0, position  1) =  0.88
similarity(position 0, position  2) =  0.64
similarity(position 0, position  5) =  0.79
similarity(position 0, position 10) =  0.42
similarity(position 0, position 19) =  0.66

=== 4. Adding position to word vectors ===
same word, no position   -> first and last vectors equal? True
same word, with position -> first and last vectors equal? False

=== 5. Why order matters ===
without position, 'bites' looks the same in both sentences? True
with position, 'bites' looks the same in both sentences?    False

=== 6. Learned positions (another common choice) ===
learned position vectors shape: (1, 3, 8)
trainable parameters: 160
sinusoidal parameters: 0 (it is a fixed formula)
```

## 🔍 Explanation
1. `sinusoidal_encoding` fills a `(max_len, d_model)` table: sine in even columns, cosine in odd columns, with different wave speeds.
2. Part 1 prints the encodings of positions 0, 1, 2 (position 0 is `[0, 1, 0, 1, ...]`).
3. Part 2 draws the whole table as text shading.
4. Part 3 shows how similar different positions are.
5. Part 4 adds the table to word vectors: the same word now looks different at different positions.
6. Part 5 uses a tiny attention: the output for 'bites' is identical in both sentences without positions, and different with them.
7. Part 6 shows the other option, a learned position embedding, and compares parameter counts.

---

## 📐 Formula and Shapes

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

```
word vectors       (batch, words, d_model)
+ position table   (words, d_model)          <- added, same table for every sentence
= input to the Transformer (batch, words, d_model)
```

| Type | Parameters | Good for |
|------|-----------|----------|
| Sinusoidal | 0 (fixed formula) | any length, no training needed |
| Learned | `max_len × d_model` | simple, common in practice |

## 📝 Important Points

- ✅ Positions are **added** to the embeddings, not joined.
- ✅ The sinusoidal table has no parameters.
- ✅ Without it, a Transformer cannot tell word order.
- ✅ A learned table only works for positions up to `max_len`.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Skipping positional encoding | The model ignores word order |
| Using a sentence longer than `max_len` | Make the table bigger |

## 🧪 Try It Yourself

- [ ] Change `d_model` to 16 and redraw the table.
- [ ] Print the encoding for position 100.
- [ ] Compare the sinusoidal and learned tables for a 5-word sentence.

## ❓ Quick Questions

1. Why do Transformers need positional encoding?
2. Is the sinusoidal encoding learned?
3. What happens to the same word at two positions after adding the encoding?

---

## 🏁 Summary

> 💡 Position vectors are added to word vectors so attention can tell word order.

⬅️ **Previous:** [Multi Head Attention](../04_multi_head_attention/multi_head_attention.md)  |  ➡️ **Next:** [Transformer Block](../06_transformer_block/transformer_block.md)
