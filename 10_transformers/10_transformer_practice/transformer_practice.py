# Practice for the Transformer section: questions, checks and interview answers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
score = 0
total = 0


def check(question, ok):
    global score, total
    total += 1
    score += ok
    print(f"Q{question}:", "correct" if ok else "wrong")


# Q1: Do attention weights in each row add up to 1?
q, k, v = torch.randn(1, 5, 8), torch.randn(1, 5, 8), torch.randn(1, 5, 8)
weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1)
print("Q1: row sums =", weights[0].sum(dim=-1).round(decimals=2).tolist())
check(1, torch.allclose(weights.sum(dim=-1), torch.ones(1, 5)))

# Q2: What shape does multi-head attention return for a (3, 10, 32) input?
mha = nn.MultiheadAttention(32, 4, batch_first=True)
out, w = mha(torch.randn(3, 10, 32), torch.randn(3, 10, 32), torch.randn(3, 10, 32))
print("Q2: output", tuple(out.shape), "| weights", tuple(w.shape))
check(2, out.shape == (3, 10, 32) and w.shape == (3, 10, 10))

# Q3: How many parameters does MultiheadAttention(d_model=16) have?  (4 * (16*16 + 16))
count = sum(p.numel() for p in nn.MultiheadAttention(16, 4).parameters())
print("Q3:", count, "= 4 * (16*16 + 16)")
check(3, count == 1088)

# Q4: What does a causal mask for 4 words look like?
causal = torch.triu(torch.ones(4, 4), diagonal=1).bool()
print("Q4:\n", causal.int())
check(4, causal[0, 1].item() is True and causal[3, 0].item() is False)

# Q5: What is the sinusoidal encoding at position 0? (sin(0) = 0, cos(0) = 1)
position = torch.tensor([[0.0]])
print("Q5: sin(0) =", torch.sin(position).item(), "| cos(0) =", torch.cos(position).item())
check(5, torch.sin(position).item() == 0.0 and torch.cos(position).item() == 1.0)

# Q6: Does a Transformer block change the shape of its input?  (No, so blocks can be stacked.)
layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, batch_first=True)
x = torch.randn(2, 7, 16)
print("Q6: in", tuple(x.shape), "-> out", tuple(layer(x).shape))
check(6, layer(x).shape == x.shape)

# Q7: How big is the attention matrix for a sentence of 100 words (one head)?
print("Q7: 100 x 100 =", 100 * 100, "numbers (it grows with the SQUARE of the length)")
check(7, 100 * 100 == 10000)

# Q8: Parameters of the feed-forward part with d_model = 16 and d_ff = 64
ffn = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16))
count = sum(p.numel() for p in ffn.parameters())
print("Q8:", count, "= (16*64 + 64) + (64*16 + 16)")
check(8, count == 2128)

# Q9: Does our attention match PyTorch's F.scaled_dot_product_attention?
ours = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1) @ v
print("Q9: same result?", torch.allclose(ours, F.scaled_dot_product_attention(q, k, v), atol=1e-5))
check(9, torch.allclose(ours, F.scaled_dot_product_attention(q, k, v), atol=1e-5))

# Q10: With a causal mask, can word 0 see word 3?  (weight must be exactly 0)
scores = torch.randn(4, 4).masked_fill(causal, float("-inf"))
masked_weights = torch.softmax(scores, dim=-1)
print("Q10: weight from word 0 to word 3 =", masked_weights[0, 3].item())
check(10, masked_weights[0, 3].item() == 0.0)

print(f"\nScore: {score}/{total}")

print("\n--- Interview style questions ---")
qa = [
    ("Why do Transformers need positional encoding?",
     "Self-attention treats the input as a set, so it has no idea about word order."),
    ("Why divide by sqrt(d) in attention?",
     "Large dot products make softmax too sharp; scaling keeps the scores in a good range."),
    ("What is the point of multiple heads?",
     "Each head can focus on a different kind of relationship at the same time."),
    ("What do residual connections and LayerNorm do?",
     "They keep training stable and let gradients flow through deep stacks."),
    ("Encoder vs decoder?",
     "An encoder looks at the whole sentence; a decoder uses a causal mask to generate left to right."),
    ("What is the main cost of attention?",
     "Time and memory grow with the square of the sequence length."),
    ("How is a Transformer different from an RNN?",
     "It processes all words in parallel instead of one after another."),
]
for i, (question, answer) in enumerate(qa, start=1):
    print(f"\nQ{i}: {question}\nA{i}: {answer}")
