# Lesson 3: Self-attention layer
# In self-attention the queries, keys and values all come from the SAME sentence.
# Three Linear layers learn how to make Q, K and V from the word vectors.

import math
import torch
import torch.nn as nn

torch.manual_seed(1)


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        return weights @ v, weights


sentence = ["the", "cat", "sat", "on", "the", "mat"]
d_model = 16
embedding = nn.Embedding(10, d_model)
ids = torch.tensor([[1, 2, 3, 4, 1, 5]])           # made-up ids for the 6 words
x = embedding(ids)                                  # (1, 6, 16)

layer = SelfAttention(d_model)
print("=== 1. Shapes ===")
out, weights = layer(x)
print("input :", tuple(x.shape), "= (batch, words, d_model)")
print("output:", tuple(out.shape))
print("weights:", tuple(weights.shape))

print("\n=== 2. Who looks at whom? (rows = word asking, columns = word looked at) ===")
print("       " + " ".join(f"{w:>5s}" for w in sentence))
for i, word in enumerate(sentence):
    row = " ".join(f"{w:5.2f}" for w in weights[0, i].tolist())
    print(f"{word:>5s}  {row}")

print("\n=== 3. The word each word pays most attention to ===")
for i, word in enumerate(sentence):
    j = weights[0, i].argmax().item()
    print(f"{word:>4s} -> {sentence[j]}")
print("(weights are random now; they become meaningful after training)")

print("\n=== 4. Parameters ===")
for name, p in layer.named_parameters():
    print(f"{name:14s} {str(tuple(p.shape)):10s} {p.numel()}")
print("total:", sum(p.numel() for p in layer.parameters()))

print("\n=== 5. Self-attention does not know word ORDER ===")
perm = torch.tensor([5, 4, 3, 2, 1, 0])             # reverse the sentence
out_rev, _ = layer(x[:, perm])
print("output of reversed sentence == reversed output?",
      torch.allclose(out_rev, out[:, perm], atol=1e-5))
print("-> we must add position information (next lessons)")
