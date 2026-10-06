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
