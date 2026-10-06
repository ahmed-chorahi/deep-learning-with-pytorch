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
