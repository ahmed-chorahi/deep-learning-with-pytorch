# 💬 Embeddings: Giving Words Meaning

> 📚 **NLP (Natural Language Processing)** &nbsp;|&nbsp; 🧩 Lesson 2 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▱▱▱` 2/5

---

## 🎯 Learning Goal
Understand what `nn.Embedding` does and how word vectors capture similarity.

## 📌 What is it?
An **embedding** is a table that gives every word a short vector of numbers. During training these vectors are learned so that words used in similar ways get similar vectors.

## 🧠 Real-Life Analogy
Imagine every word is a point on a map. Words with similar meaning live close together (good and great), and opposite words live far apart (good and bad).

---

## 💻 Example

Code from `embeddings.py` (run it with `python embeddings.py`):

```python
# Lesson 2: Embeddings
# An embedding gives every word a small vector of numbers.
# Words with similar meaning should end up with similar vectors.

import torch
import torch.nn as nn

torch.manual_seed(0)

vocab = {"<pad>": 0, "good": 1, "great": 2, "bad": 3, "awful": 4, "movie": 5}
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4, padding_idx=0)

print("=== 1. The embedding table ===")
print("table shape:", tuple(embedding.weight.shape), "= (words, vector size)")
print("vector for 'good':", embedding.weight[vocab["good"]].data.round(decimals=2))

print("\n=== 2. Looking up words ===")
ids = torch.tensor([1, 3, 5])                  # good, bad, movie
vectors = embedding(ids)
print("input ids shape :", tuple(ids.shape))
print("output shape    :", tuple(vectors.shape))

print("\n=== 3. A batch of sentences ===")
batch = torch.tensor([[1, 5, 0],               # "good movie <pad>"
                      [3, 5, 0]])              # "bad movie <pad>"
out = embedding(batch)
print("batch shape :", tuple(batch.shape), "= (sentences, words)")
print("output shape:", tuple(out.shape), "= (sentences, words, vector size)")
print("the <pad> vector is all zeros:", out[0, 2].data)

print("\n=== 4. One vector per sentence (average of its words) ===")
sentence_vectors = out.mean(dim=1)
print("sentence vectors shape:", tuple(sentence_vectors.shape))

print("\n=== 5. Similarity of words (cosine similarity) ===")
# Embeddings start random. Here we set meaningful vectors by hand to show the idea.
with torch.no_grad():
    embedding.weight[vocab["good"]] = torch.tensor([0.9, 0.8, 0.1, 0.0])
    embedding.weight[vocab["great"]] = torch.tensor([1.0, 0.9, 0.0, 0.1])
    embedding.weight[vocab["bad"]] = torch.tensor([-0.9, -0.8, 0.1, 0.0])
    embedding.weight[vocab["awful"]] = torch.tensor([-1.0, -0.9, 0.0, 0.1])


def similarity(w1, w2):
    v1 = embedding.weight[vocab[w1]]
    v2 = embedding.weight[vocab[w2]]
    return torch.cosine_similarity(v1, v2, dim=0).item()


for a, b in [("good", "great"), ("bad", "awful"), ("good", "bad")]:
    print(f"similarity({a}, {b}) = {similarity(a, b):5.2f}")

print("\n=== 6. Embeddings are learnable parameters ===")
print("number of parameters:", embedding.weight.numel(), "= 6 words * 4 numbers")
print("requires_grad:", embedding.weight.requires_grad)
```

## 📤 Expected Output

```text
=== 1. The embedding table ===
table shape: (6, 4) = (words, vector size)
vector for 'good': tensor([ 0.8500,  0.6900, -0.3200, -2.1200])

=== 2. Looking up words ===
input ids shape : (3,)
output shape    : (3, 4)

=== 3. A batch of sentences ===
batch shape : (2, 3) = (sentences, words)
output shape: (2, 3, 4) = (sentences, words, vector size)
the <pad> vector is all zeros: tensor([0., 0., 0., 0.])

=== 4. One vector per sentence (average of its words) ===
sentence vectors shape: (2, 4)

=== 5. Similarity of words (cosine similarity) ===
similarity(good, great) =  0.99
similarity(bad, awful) =  0.99
similarity(good, bad) = -0.99

=== 6. Embeddings are learnable parameters ===
number of parameters: 24 = 6 words * 4 numbers
requires_grad: True
```

## 🔍 Explanation
1. Part 1 creates an embedding table of shape `(6 words, 4 numbers)`.
2. Part 2 looks up three ids and gets three vectors.
3. Part 3 looks up a batch of sentences: `(2, 3)` ids become `(2, 3, 4)` vectors.
4. Part 4 averages the word vectors to get one vector per sentence.
5. Part 5 sets meaningful vectors by hand and measures cosine similarity.
6. Part 6 shows the embedding weights are learnable parameters.

---

## 📐 Embedding Shapes

```
ids    : (batch, words)                  e.g. (2, 3)
              │  nn.Embedding(vocab, 4)
              ▼
vectors: (batch, words, embedding_dim)   e.g. (2, 3, 4)
              │  mean(dim=1)
              ▼
sentence vectors: (batch, embedding_dim) e.g. (2, 4)
```

**Cosine similarity**: 1 = same direction, 0 = unrelated, -1 = opposite.

## 📝 Important Points

- ✅ An embedding is just a lookup table that can be trained.
- ✅ `padding_idx=0` keeps the `<pad>` vector at zero.
- ✅ Random embeddings only become meaningful after training.
- ✅ Cosine similarity compares directions of vectors.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Passing floats to `nn.Embedding` | Ids must be integers (`long`) |
| Ids bigger than the vocabulary size | Keep ids below `num_embeddings` |

## 🧪 Try It Yourself

- [ ] Change `embedding_dim` to 8 and check the output shape.
- [ ] Add a word to the vocabulary and use it.
- [ ] Compute similarity between `great` and `awful`.

## ❓ Quick Questions

1. What shape does the embedding output have for a batch of ids?
2. Why use embeddings instead of one-hot vectors?
3. What does a cosine similarity of -1 mean?

---

## 🏁 Summary

> 💡 An embedding turns each word id into a learnable vector.

⬅️ **Previous:** [Tokenization](../01_tokenization/tokenization.md)  |  ➡️ **Next:** [Bag Of Words](../03_bag_of_words/bag_of_words.md)
