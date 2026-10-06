# 🤖 Multi-Head Attention: Many Views at Once

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 4 of 10 &nbsp;|&nbsp; ⏱️ ~35 min &nbsp;|&nbsp; 🎓 🟠 Upper-Intermediate
>
> Progress: `▰▰▰▰▱▱▱▱▱▱` 4/10

---

## 🎯 Learning Goal
Build multi-head attention from scratch and check that it matches PyTorch's own layer.

## 📌 What is it?
**Multi-head attention** splits the vectors into several smaller pieces (heads). Each head runs its own attention, so different heads can learn different relationships. The results are joined back together and mixed by a final Linear layer.

## 🧠 Real-Life Analogy
Several people read the same sentence at once: one looks at grammar, one at meaning, one at nearby words. At the end they share what they found.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Head** | one independent attention operation |
| **head_dim** | `d_model // num_heads` |
| **split_heads** | reshape `(batch, words, d_model)` to `(batch, heads, words, head_dim)` |
| **Output projection** | a Linear layer that mixes the joined heads |

---

## 💻 Example

Code from `multi_head_attention.py` (run it with `python multi_head_attention.py`):

```python
# Lesson 4: Multi-head attention
# Instead of one attention, we run several smaller ones ("heads") in parallel.
# Each head can learn a different kind of relationship (grammar, meaning, position...).

import math
import torch
import torch.nn as nn

torch.manual_seed(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # (batch, words, d_model) -> (batch, heads, words, head_dim)
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, show_shapes=False):
        q = self.split_heads(self.q(x))
        k = self.split_heads(self.k(x))
        v = self.split_heads(self.v(x))
        if show_shapes:
            print("after split_heads :", tuple(q.shape), "= (batch, heads, words, head_dim)")
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        if show_shapes:
            print("attention weights :", tuple(weights.shape), "= (batch, heads, words, words)")
        out = weights @ v
        # (batch, heads, words, head_dim) -> (batch, words, d_model)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        if show_shapes:
            print("after joining heads:", tuple(out.shape))
        return self.out(out), weights


d_model, num_heads = 16, 4
x = torch.randn(2, 5, d_model)                      # 2 sentences, 5 words each
mha = MultiHeadAttention(d_model, num_heads)

print("=== 1. Shapes step by step ===")
print("input             :", tuple(x.shape))
out, weights = mha(x, show_shapes=True)
print("final output      :", tuple(out.shape))

print("\n=== 2. Each head has its own attention pattern (word 0 of sentence 0) ===")
for h in range(num_heads):
    print(f"head {h}:", weights[0, h, 0].detach().round(decimals=2).tolist())

print("\n=== 3. Size of each head ===")
print(f"d_model = {d_model}, heads = {num_heads} -> head_dim = {d_model // num_heads}")

print("\n=== 4. Parameters ===")
total = sum(p.numel() for p in mha.parameters())
print("our MultiHeadAttention:", total, "= 4 * (16*16 + 16)")

print("\n=== 5. Compare with PyTorch's nn.MultiheadAttention ===")
builtin = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
print("built-in parameters   :", sum(p.numel() for p in builtin.parameters()))

# copy our weights into the built-in layer; the answers should then match
with torch.no_grad():
    builtin.in_proj_weight.copy_(torch.cat([mha.q.weight, mha.k.weight, mha.v.weight]))
    builtin.in_proj_bias.copy_(torch.cat([mha.q.bias, mha.k.bias, mha.v.bias]))
    builtin.out_proj.weight.copy_(mha.out.weight)
    builtin.out_proj.bias.copy_(mha.out.bias)
builtin.eval()
builtin_out, _ = builtin(x, x, x)
print("same output as ours?  :", torch.allclose(out, builtin_out, atol=1e-5))

print("\n=== 6. Try other head counts ===")
for heads in [1, 2, 8]:
    layer = MultiHeadAttention(d_model, heads)
    print(f"heads = {heads}: output {tuple(layer(x)[0].shape)}, parameters {sum(p.numel() for p in layer.parameters())}")
```

## 📤 Expected Output

```text
=== 1. Shapes step by step ===
input             : (2, 5, 16)
after split_heads : (2, 4, 5, 4) = (batch, heads, words, head_dim)
attention weights : (2, 4, 5, 5) = (batch, heads, words, words)
after joining heads: (2, 5, 16)
final output      : (2, 5, 16)

=== 2. Each head has its own attention pattern (word 0 of sentence 0) ===
head 0: [0.1899999976158142, 0.23999999463558197, 0.18000000715255737, 0.1899999976158142, 0.20000000298023224]
head 1: [0.3199999928474426, 0.11999999731779099, 0.20999999344348907, 0.20000000298023224, 0.15000000596046448]
head 2: [0.15000000596046448, 0.07999999821186066, 0.07000000029802322, 0.1899999976158142, 0.5099999904632568]
head 3: [0.1599999964237213, 0.1899999976158142, 0.10000000149011612, 0.23999999463558197, 0.30000001192092896]

=== 3. Size of each head ===
d_model = 16, heads = 4 -> head_dim = 4

=== 4. Parameters ===
our MultiHeadAttention: 1088 = 4 * (16*16 + 16)

=== 5. Compare with PyTorch's nn.MultiheadAttention ===
built-in parameters   : 1088
same output as ours?  : True

=== 6. Try other head counts ===
heads = 1: output (2, 5, 16), parameters 1088
heads = 2: output (2, 5, 16), parameters 1088
heads = 8: output (2, 5, 16), parameters 1088
```

## 🔍 Explanation
1. `MultiHeadAttention(d_model=16, num_heads=4)` has four Linear layers: q, k, v and out.
2. `split_heads` uses `view` and `transpose` to create the head dimension.
3. Attention runs on all heads at once; the weights are `(batch, heads, words, words)`.
4. Heads are joined with `transpose`, `contiguous` and `view`, then the output layer is applied.
5. Part 1 prints the shape after each step.
6. Part 2 shows each head's attention for the first word: they differ.
7. Part 4 counts parameters: 4 × (16×16 + 16) = 1088.
8. Part 5 copies our weights into `nn.MultiheadAttention`; both give the same output.
9. Part 6 tries 1, 2 and 8 heads; the parameter count stays the same.

---

## 📐 Splitting into Heads

```
x                         (batch, words, 16)
 │  Linear (q, k, v)
 ▼
split into 4 heads        (batch, 4, words, 4)      head_dim = 16 / 4 = 4
 │  attention in each head
 ▼
weights                   (batch, 4, words, words)
output per head           (batch, 4, words, 4)
 │  join heads
 ▼
(batch, words, 16)
 │  Linear (out)
 ▼
final output              (batch, words, 16)
```

| Heads | head_dim | Parameters |
|-------|----------|------------|
| 1 | 16 | 1088 |
| 4 | 4 | 1088 |
| 8 | 2 | 1088 |

More heads do **not** add parameters; they just divide the same vector into more pieces.

## 📝 Important Points

- ✅ `d_model` must be divisible by `num_heads`.
- ✅ The shape in and out is the same: `(batch, words, d_model)`.
- ✅ Each head can learn a different pattern.
- ✅ Our version matches PyTorch's layer after copying the weights.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| `d_model` not divisible by the number of heads | Pick 16/4, 32/8, 64/8... |
| Using `view` after `transpose` without `contiguous` | Call `.contiguous()` first |

## 🧪 Try It Yourself

- [ ] Try `d_model = 32` and 8 heads.
- [ ] Print which head pays the most attention to word 0.
- [ ] Add an attention mask to the forward function.

## ❓ Quick Questions

1. Why use several heads instead of one?
2. What is `head_dim`?
3. Does using more heads increase the number of parameters?

---

## 🏁 Summary

> 💡 Multi-head attention = several small attentions in parallel, joined and mixed.

⬅️ **Previous:** [Self Attention](../03_self_attention/self_attention.md)  |  ➡️ **Next:** [Positional Encoding](../05_positional_encoding/positional_encoding.md)
