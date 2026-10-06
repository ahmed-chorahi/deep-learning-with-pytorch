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
