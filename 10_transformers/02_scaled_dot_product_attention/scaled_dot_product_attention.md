# 🤖 Scaled Dot-Product Attention: The Formula

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 2 of 10 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▱▱▱▱▱▱▱▱` 2/10

---

## 🎯 Learning Goal
Write the Transformer attention formula as a PyTorch function and understand every part, including masks.

## 📌 What is it?
**Scaled dot-product attention** is `softmax(Q·Kᵀ / √d) · V`. Q, K and V are three tensors of the same shape `(batch, words, d)`. Dividing by `√d` stops the scores from growing too large, and **masks** hide positions we must not look at.

## 🧠 Real-Life Analogy
Like a group chat where every person (word) asks a question (Q), everyone shows a name tag (K), and what they say is the content (V). The `√d` is a volume knob that stops the loudest voices from drowning out everyone else.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Q, K, V** | queries, keys and values |
| **d** | size of each vector |
| **Scaling** | dividing the scores by `√d` |
| **Padding mask** | hides `<pad>` positions |
| **Causal mask** | hides future words |

---

## 💻 Example

Code from `scaled_dot_product_attention.py` (run it with `python scaled_dot_product_attention.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. Shapes ===
q, k, v : (1, 4, 8) = (batch, words, d)
weights : (1, 4, 4) = (batch, words, words)
output  : (1, 4, 8) = same shape as v

=== 2. Every row of weights sums to 1 ===
tensor([[0.1200, 0.7400, 0.1100, 0.0300],
        [0.2600, 0.2700, 0.1700, 0.3100],
        [0.3400, 0.0200, 0.1900, 0.4600],
        [0.2000, 0.6200, 0.1000, 0.0700]])
row sums: [1.0, 1.0, 1.0, 1.0]

=== 3. Why divide by sqrt(d)? ===
d =   4 | std of dot products: raw =   1.9, scaled = 1.0
d =  64 | std of dot products: raw =   8.1, scaled = 1.0
d = 512 | std of dot products: raw =  22.6, scaled = 1.0
Without scaling, big d gives huge scores -> softmax becomes 'all or nothing'.

=== 4. Padding mask (ignore <pad> words) ===
weights given to the padded word (last column): [0.0, 0.0, 0.0, 0.0]

=== 5. Causal mask (a word may only look at earlier words) ===
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4900, 0.5100, 0.0000, 0.0000],
        [0.6200, 0.0300, 0.3500, 0.0000],
        [0.2000, 0.6200, 0.1000, 0.0700]])

=== 6. Compare with PyTorch's own function ===
same output as ours? True
```

## 🔍 Explanation
1. `attention(q, k, v, mask)` computes the scores `q @ k.transpose(-2, -1) / sqrt(d)`.
2. `masked_fill(mask == 0, -inf)` hides positions; softmax turns `-inf` into weight 0.
3. Part 1 prints the shapes: weights are `(batch, words, words)`, the output has the shape of `v`.
4. Part 2 shows every row of the weights sums to 1.
5. Part 3 measures the spread of dot products for d = 4, 64, 512: raw scores grow with d, scaled scores stay near 1.
6. Part 4 hides a padded word: its weight is exactly 0.
7. Part 5 uses a lower-triangular matrix as a causal mask.
8. Part 6 checks our function against `F.scaled_dot_product_attention`.

---

## 📐 Shapes and Masks

```
Q  (batch, words, d)  ─┐
                       ├─► Q @ Kᵀ ─► (batch, words, words) ─► ÷ √d ─► mask ─► softmax ─► @ V
K  (batch, words, d)  ─┘                                                            │
V  (batch, words, d) ────────────────────────────────────────────────────────────────┘
                                                              output (batch, words, d)
```

**Causal mask for 4 words** (1 = allowed to look):

```
1 0 0 0      word 1 sees only word 1
1 1 0 0      word 2 sees words 1-2
1 1 1 0      word 3 sees words 1-3
1 1 1 1      word 4 sees everything before it
```

## 📝 Important Points

- ✅ The scores matrix is `words × words`, so cost grows with the square of the length.
- ✅ Scaling keeps softmax from becoming 'all or nothing'.
- ✅ Masked positions get `-inf` before softmax, so their weight is exactly 0.
- ✅ `F.scaled_dot_product_attention` is PyTorch's built-in version.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting to divide by `√d` | Training becomes unstable for large `d` |
| Applying the mask after softmax | Mask the scores *before* softmax |
| Transposing the wrong dimensions | Use `k.transpose(-2, -1)` |

## 🧪 Try It Yourself

- [ ] Remove the scaling and print the weights for d = 512.
- [ ] Mask the first word instead of the last.
- [ ] Use a batch of 3 sentences.

## ❓ Quick Questions

1. Why do we divide by √d?
2. What shape is the attention weight matrix?
3. Why is `-inf` used for masking?

---

## 🏁 Summary

> 💡 softmax(Q·Kᵀ/√d)·V, with masks to hide padding or the future.

⬅️ **Previous:** [Attention Intuition](../01_attention_intuition/attention_intuition.md)  |  ➡️ **Next:** [Self Attention](../03_self_attention/self_attention.md)
