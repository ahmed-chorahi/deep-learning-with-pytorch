# Lesson 2: Scaled dot-product attention (the heart of the Transformer)
#   Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
# Q = queries (what I look for), K = keys (what I offer), V = values (what I give).

import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)      # (batch, seq, seq)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))   # hide masked positions
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


batch, seq, d = 1, 4, 8
q = torch.randn(batch, seq, d)
k = torch.randn(batch, seq, d)
v = torch.randn(batch, seq, d)

print("=== 1. Shapes ===")
out, weights = attention(q, k, v)
print("q, k, v :", tuple(q.shape), "= (batch, words, d)")
print("weights :", tuple(weights.shape), "= (batch, words, words)")
print("output  :", tuple(out.shape), "= same shape as v")

print("\n=== 2. Every row of weights sums to 1 ===")
print(weights[0].round(decimals=2))
print("row sums:", weights[0].sum(dim=-1).round(decimals=2).tolist())

print("\n=== 3. Why divide by sqrt(d)? ===")
for d_big in [4, 64, 512]:
    a = torch.randn(1000, d_big)
    b = torch.randn(1000, d_big)
    raw = (a * b).sum(dim=1)                             # dot products
    print(f"d = {d_big:3d} | std of dot products: raw = {raw.std().item():5.1f}"
          f", scaled = {(raw / math.sqrt(d_big)).std().item():.1f}")
print("Without scaling, big d gives huge scores -> softmax becomes 'all or nothing'.")

print("\n=== 4. Padding mask (ignore <pad> words) ===")
mask = torch.tensor([[[1, 1, 1, 0]] * 4])                # last word is padding
out_m, weights_m = attention(q, k, v, mask)
print("weights given to the padded word (last column):", weights_m[0, :, -1].tolist())

print("\n=== 5. Causal mask (a word may only look at earlier words) ===")
causal = torch.tril(torch.ones(seq, seq)).unsqueeze(0)   # lower triangle of ones
print(causal[0])
out_c, weights_c = attention(q, k, v, causal)
print(weights_c[0].round(decimals=2))

print("\n=== 6. Compare with PyTorch's own function ===")
builtin = F.scaled_dot_product_attention(q, k, v)
print("same output as ours?", torch.allclose(out, builtin, atol=1e-5))
