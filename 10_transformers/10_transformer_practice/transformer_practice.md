# 🤖 Transformer Practice and Interview Questions

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 10 of 10 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▰▰▰▰▰▰▰▰▰` 10/10

---

## 🎯 Learning Goal
Test your understanding with ten checked questions and seven interview-style answers.

## 📌 What is it?
Each question runs a small piece of code and checks the answer automatically, then prints a score. At the end you get a list of typical Transformer interview questions with short answers.

## 🧠 Real-Life Analogy
Like the end-of-chapter exam: the earlier lessons taught you the tools, this lesson checks you can use them.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Attention weights** | each row sums to 1 |
| **Causal mask** | upper triangle hidden |
| **d_ff** | feed-forward hidden size |
| **Quadratic cost** | attention memory grows with the square of the length |

---

## 💻 Example

Code from `transformer_practice.py` (run it with `python transformer_practice.py`):

```python
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
```

## 📤 Expected Output

```text
Q1: row sums = [1.0, 1.0, 1.0, 1.0, 1.0]
Q1: correct
Q2: output (3, 10, 32) | weights (3, 10, 10)
Q2: correct
Q3: 1088 = 4 * (16*16 + 16)
Q3: correct
Q4:
 tensor([[0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]], dtype=torch.int32)
Q4: correct
Q5: sin(0) = 0.0 | cos(0) = 1.0
Q5: correct
Q6: in (2, 7, 16) -> out (2, 7, 16)
Q6: correct
Q7: 100 x 100 = 10000 numbers (it grows with the SQUARE of the length)
Q7: correct
Q8: 2128 = (16*64 + 64) + (64*16 + 16)
Q8: correct
Q9: same result? True
Q9: correct
Q10: weight from word 0 to word 3 = 0.0
Q10: correct

Score: 10/10

--- Interview style questions ---

Q1: Why do Transformers need positional encoding?
A1: Self-attention treats the input as a set, so it has no idea about word order.

Q2: Why divide by sqrt(d) in attention?
A2: Large dot products make softmax too sharp; scaling keeps the scores in a good range.

Q3: What is the point of multiple heads?
A3: Each head can focus on a different kind of relationship at the same time.

Q4: What do residual connections and LayerNorm do?
A4: They keep training stable and let gradients flow through deep stacks.

Q5: Encoder vs decoder?
A5: An encoder looks at the whole sentence; a decoder uses a causal mask to generate left to right.

Q6: What is the main cost of attention?
A6: Time and memory grow with the square of the sequence length.

Q7: How is a Transformer different from an RNN?
A7: It processes all words in parallel instead of one after another.
```

## 🔍 Explanation
1. Q1: attention weights in each row add up to 1.
2. Q2: shapes returned by `nn.MultiheadAttention`.
3. Q3: parameter count of multi-head attention (1088 for `d_model=16`).
4. Q4: what a causal mask looks like.
5. Q5: sinusoidal encoding at position 0.
6. Q6: a Transformer layer does not change the shape.
7. Q7: the attention matrix for 100 words has 10,000 numbers.
8. Q8: feed-forward parameters (2128).
9. Q9: our attention equals `F.scaled_dot_product_attention`.
10. Q10: with a causal mask, word 0 gives weight 0 to word 3.
11. The last part prints seven interview questions with answers.

---

## 📐 Formulas to Remember

| Item | Formula | Example |
|------|---------|---------|
| Attention | `softmax(QKᵀ/√d)·V` | |
| Multi-head attention params | `4 × (d² + d)` | d=16 → 1088 |
| Feed-forward params | `(d·d_ff + d_ff) + (d_ff·d + d)` | 16, 64 → 2128 |
| Attention matrix size | `words²` | 100 words → 10,000 |
| head_dim | `d_model / heads` | 16 / 4 = 4 |

```
Encoder block:  x → Attention → Add&Norm → FeedForward → Add&Norm
Decoder-only:   same, but with a causal mask (GPT)
```

## 📝 Important Points

- ✅ Try each question before reading the code.
- ✅ Know the shapes at every step of a Transformer.
- ✅ Be able to explain *why* each part exists, not only what it does.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Memorizing answers | Run the code and explain it in your own words |

## 🧪 Try It Yourself

- [ ] Write 3 more questions with checks.
- [ ] Explain multi-head attention to a friend in two sentences.
- [ ] Draw a Transformer block on paper from memory.

## ❓ Quick Questions

1. What is the shape of the attention weights?
2. How many parameters does multi-head attention have for `d_model=32`?
3. What is the difference between an encoder and a decoder?

---

## 🏁 Summary

> 💡 If you can answer these, you understand the core of the Transformer.

⬅️ **Previous:** [Causal Language Model](../09_causal_language_model/causal_language_model.md)
