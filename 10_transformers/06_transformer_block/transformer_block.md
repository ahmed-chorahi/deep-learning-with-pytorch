# 🤖 The Transformer Block: Attention + Feed-Forward

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 6 of 10 &nbsp;|&nbsp; ⏱️ ~35 min &nbsp;|&nbsp; 🎓 🟠 Upper-Intermediate
>
> Progress: `▰▰▰▰▰▰▱▱▱▱` 6/10

---

## 🎯 Learning Goal
Assemble one complete Transformer block with residual connections and layer normalization.

## 📌 What is it?
A **Transformer block** has two parts: multi-head attention (words talk to each other) and a feed-forward network (each word is processed on its own). Both are wrapped with a **residual connection** (add the input back) and **LayerNorm**.

## 🧠 Real-Life Analogy
A team meeting followed by individual work: first everyone discusses (attention), then each person thinks alone (feed-forward). After each step, they keep their original notes (residual) and tidy everything onto the same scale (LayerNorm).

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Residual connection** | `x + sublayer(x)`: add the input to the output |
| **LayerNorm** | normalizes each word vector to mean 0, std 1 |
| **Feed-forward network** | Linear → ReLU → Linear applied to every word separately |
| **d_ff** | hidden size of the feed-forward network, often `4 × d_model` |
| **Dropout** | randomly switches off values to reduce overfitting |

---

## 💻 Example

Code from `transformer_block.py` (run it with `python transformer_block.py`):

```python
# Lesson 6: One Transformer block
#   x -> Multi-head attention -> Add & Norm -> Feed-forward -> Add & Norm
# "Add" = residual connection (add the input back). "Norm" = LayerNorm.

import torch
import torch.nn as nn

torch.manual_seed(0)

print("=== 1. LayerNorm: normalize each word vector on its own ===")
x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [10.0, 20.0, 30.0, 40.0]])
norm = nn.LayerNorm(4)
y = norm(x)
print("mean of each row:", y.mean(dim=1).detach().round(decimals=3).tolist())
print("std  of each row:", y.std(dim=1, unbiased=False).detach().round(decimals=3).tolist())

print("\n=== 2. Feed-forward network: grow, ReLU, shrink ===")
d_model, d_ff = 16, 64
ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
word_vectors = torch.randn(2, 5, d_model)
print("input :", tuple(word_vectors.shape))
print("output:", tuple(ffn(word_vectors).shape), "(same shape; applied to each word separately)")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # (we built our own attention in lesson 4; here we use PyTorch's version)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))      # residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))       # residual + norm
        return x


block = TransformerBlock(d_model=16, num_heads=4, d_ff=64)
print("\n=== 3. The block ===")
print(block)

print("\n=== 4. Shapes stay the same, so blocks can be stacked ===")
x = torch.randn(2, 5, 16)
print("input :", tuple(x.shape))
print("output:", tuple(block(x).shape))
stack = nn.Sequential(*[TransformerBlock(16, 4, 64) for _ in range(3)])
print("after 3 stacked blocks:", tuple(stack(x).shape))

print("\n=== 5. Parameters inside one block ===")
total = 0
for name, p in block.named_parameters():
    total += p.numel()
    print(f"{name:28s} {p.numel():5d}")
print("total:", total)

print("\n=== 6. Why residual connections? ===")
block.eval()
out = block(x)
print("how far the output moved from the input (mean abs difference):", round((out - x).abs().mean().item(), 3))
print("Residuals let information and gradients flow straight through deep stacks.")
```

## 📤 Expected Output

```text
=== 1. LayerNorm: normalize each word vector on its own ===
mean of each row: [0.0, 0.0]
std  of each row: [1.0, 1.0]

=== 2. Feed-forward network: grow, ReLU, shrink ===
input : (2, 5, 16)
output: (2, 5, 16) (same shape; applied to each word separately)

=== 3. The block ===
TransformerBlock(
  (attention): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=16, out_features=16, bias=True)
  )
  (norm1): LayerNorm((16,), eps=1e-05, elementwise_affine=True, bias=True)
  (ffn): Sequential(
    (0): Linear(in_features=16, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=16, bias=True)
  )
  (norm2): LayerNorm((16,), eps=1e-05, elementwise_affine=True, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)

=== 4. Shapes stay the same, so blocks can be stacked ===
input : (2, 5, 16)
output: (2, 5, 16)
after 3 stacked blocks: (2, 5, 16)

=== 5. Parameters inside one block ===
attention.in_proj_weight       768
attention.in_proj_bias          48
attention.out_proj.weight      256
attention.out_proj.bias         16
norm1.weight                    16
norm1.bias                      16
ffn.0.weight                  1024
ffn.0.bias                      64
ffn.2.weight                  1024
ffn.2.bias                      16
norm2.weight                    16
norm2.bias                      16
total: 3280

=== 6. Why residual connections? ===
how far the output moved from the input (mean abs difference): 0.323
Residuals let information and gradients flow straight through deep stacks.
```

## 🔍 Explanation
1. Part 1: `LayerNorm` makes each row have mean 0 and std 1, even when the numbers are 10 times bigger.
2. Part 2: the feed-forward network grows 16 → 64, applies ReLU and shrinks back to 16, so the shape is unchanged.
3. `TransformerBlock.forward` does attention, then `norm1(x + attn_out)`, then the feed-forward network and `norm2(x + ffn_out)`.
4. Part 3 prints the whole block.
5. Part 4 shows the shape stays `(2, 5, 16)`, so blocks can be stacked in `nn.Sequential`.
6. Part 5 lists every parameter: 3280 in total for `d_model=16`, `d_ff=64`.
7. Part 6 measures how far the output moves from the input to show the residual path.

---

## 📐 Inside One Block

```
input x  (batch, words, d_model)
   │
   ├───────────────┐
   ▼               │
Multi-Head Attention
   │               │
   ▼               │
  Add  ◄───────────┘        (residual)
   │
LayerNorm
   │
   ├───────────────┐
   ▼               │
Feed-Forward (d_model → d_ff → d_model)
   │               │
   ▼               │
  Add  ◄───────────┘        (residual)
   │
LayerNorm
   ▼
output  (batch, words, d_model)     same shape as the input!
```

## 📝 Important Points

- ✅ The output has the same shape as the input, so you can stack many blocks.
- ✅ Attention mixes information *between* words; the feed-forward network works on each word separately.
- ✅ Residual connections help gradients flow in deep networks.
- ✅ This version is 'post-norm' (norm after the residual), like the original paper.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Forgetting the residual connection | Deep stacks become hard to train |
| Wrong `d_model`/heads combination | `d_model` must be divisible by the heads |

## 🧪 Try It Yourself

- [ ] Stack 6 blocks and count the parameters.
- [ ] Change `d_ff` to 128.
- [ ] Set `dropout=0` and compare outputs.

## ❓ Quick Questions

1. What does LayerNorm do?
2. Why is the output shape equal to the input shape?
3. What is the job of the feed-forward network?

---

## 🏁 Summary

> 💡 Attention + feed-forward, each with a residual connection and LayerNorm, is one Transformer block.

⬅️ **Previous:** [Positional Encoding](../05_positional_encoding/positional_encoding.md)  |  ➡️ **Next:** [Transformer Encoder](../07_transformer_encoder/transformer_encoder.md)
