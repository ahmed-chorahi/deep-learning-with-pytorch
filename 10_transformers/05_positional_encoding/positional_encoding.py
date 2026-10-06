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
