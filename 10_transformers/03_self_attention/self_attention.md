# 🤖 Self-Attention: A Sentence Looks at Itself

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 3 of 10 &nbsp;|&nbsp; ⏱️ ~30 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▱▱▱▱▱▱▱` 3/10

---

## 🎯 Learning Goal
Turn the attention formula into a PyTorch layer with learnable Q, K and V.

## 📌 What is it?
In **self-attention** the queries, keys and values all come from the same sentence. Three `nn.Linear` layers learn how to create Q, K and V from the word vectors, so the model can learn *what to look for* and *what to offer*.

## 🧠 Real-Life Analogy
Every student in a class (word) writes a question, a name tag and a note. Then everyone reads the notes of the others, paying more attention to the students whose name tags match their question.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Self-attention** | attention where Q, K and V come from the same input |
| **Projection** | a Linear layer that creates Q, K or V |
| **Attention map** | the weights matrix: who looks at whom |
| **Permutation equivariant** | reordering the input reorders the output the same way |

---

## 💻 Example

Code from `self_attention.py` (run it with `python self_attention.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. Shapes ===
input : (1, 6, 16) = (batch, words, d_model)
output: (1, 6, 16)
weights: (1, 6, 6)

=== 2. Who looks at whom? (rows = word asking, columns = word looked at) ===
         the   cat   sat    on   the   mat
  the   0.15  0.22  0.15  0.19  0.15  0.13
  cat   0.18  0.13  0.12  0.20  0.18  0.19
  sat   0.16  0.20  0.18  0.16  0.16  0.15
   on   0.18  0.17  0.16  0.18  0.18  0.13
  the   0.15  0.22  0.15  0.19  0.15  0.13
  mat   0.18  0.14  0.22  0.16  0.18  0.12

=== 3. The word each word pays most attention to ===
 the -> cat
 cat -> on
 sat -> cat
  on -> the
 the -> cat
 mat -> sat
(weights are random now; they become meaningful after training)

=== 4. Parameters ===
q_proj.weight  (16, 16)   256
q_proj.bias    (16,)      16
k_proj.weight  (16, 16)   256
k_proj.bias    (16,)      16
v_proj.weight  (16, 16)   256
v_proj.bias    (16,)      16
total: 816

=== 5. Self-attention does not know word ORDER ===
output of reversed sentence == reversed output? True
-> we must add position information (next lessons)
```

## 🔍 Explanation
1. `SelfAttention` has `q_proj`, `k_proj` and `v_proj` (`nn.Linear(d_model, d_model)`).
2. The forward pass makes Q, K, V, computes the scaled scores, softmax and the weighted sum.
3. Part 1: for the sentence 'the cat sat on the mat' the input is `(1, 6, 16)` and the weights are `(1, 6, 6)`.
4. Part 2 prints the attention map as a table: each row sums to 1.
5. Part 3 lists the word each word looks at most; the weights are random until the model is trained.
6. Part 4 counts parameters: 3 × (16×16 + 16) = 816.
7. Part 5 reverses the sentence: the output is just the reversed output, so self-attention cannot tell word order.

---

## 📐 The Attention Map

```
            the   cat   sat    on   the   mat      <- words being looked at
 the      [ .17   .15   .18   .16   .17   .17 ]      each ROW sums to 1
 cat      [ ...                                ]
 sat      [ ...                                ]
 ...
```

| Part | Shape |
|------|-------|
| input `x` | `(batch, words, d_model)` |
| Q, K, V | `(batch, words, d_model)` |
| weights | `(batch, words, words)` |
| output | `(batch, words, d_model)` |

The numbers in the map above are only an illustration; your run will show different values.

## 📝 Important Points

- ✅ Self-attention gives every word a view of the *whole* sentence in one step.
- ✅ All words are processed in parallel (unlike an RNN).
- ✅ It has no idea about word order on its own.
- ✅ The weights are random at the start and become meaningful after training.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Expecting meaningful attention before training | Weights are random until the model learns |
| Forgetting word order is missing | Add positional encoding (lesson 5) |

## 🧪 Try It Yourself

- [ ] Print the attention map for a sentence of 4 words.
- [ ] Change `d_model` to 32 and count the parameters.
- [ ] Add an output `nn.Linear` after the weighted sum.

## ❓ Quick Questions

1. What is the difference between attention and self-attention?
2. How many parameters do three `Linear(16, 16)` layers have?
3. Why can't self-attention tell the order of words?

---

## 🏁 Summary

> 💡 Self-attention lets each word gather information from all other words using learned Q, K and V.

⬅️ **Previous:** [Scaled Dot Product Attention](../02_scaled_dot_product_attention/scaled_dot_product_attention.md)  |  ➡️ **Next:** [Multi Head Attention](../04_multi_head_attention/multi_head_attention.md)
