# Lesson 1: Attention intuition
# Attention answers: "Which other words should I look at, and how much?"
# Steps: compare (dot product) -> turn scores into weights (softmax) -> mix the values.

import torch

# Each word gets a tiny hand-made vector: [animal-ness, vehicle-ness, food-ness]
words = ["cat", "dog", "car", "apple"]
vectors = torch.tensor([[0.9, 0.1, 0.0],     # cat
                        [0.8, 0.2, 0.0],     # dog
                        [0.1, 0.9, 0.0],     # car
                        [0.0, 0.1, 0.9]])    # apple

print("=== 1. A query: 'I am looking for animals' ===")
query = torch.tensor([1.0, 0.0, 0.0])
scores = vectors @ query                      # dot product = how well each word matches
print("scores :", dict(zip(words, scores.tolist())))

print("\n=== 2. Softmax turns scores into weights that add up to 1 ===")
weights = torch.softmax(scores, dim=0)
for word, w in zip(words, weights):
    print(f"{word:6s} {w.item():.2f} {'#' * int(w.item() * 40)}")
print("sum of weights:", round(weights.sum().item(), 2))

print("\n=== 3. The output is a weighted mix of the word vectors ===")
output = weights @ vectors                    # (4,) @ (4, 3) -> (3,)
print("output:", output.round(decimals=2))
print("-> mostly 'animal', because cat and dog got the biggest weights")

print("\n=== 4. A different query: 'I am looking for vehicles' ===")
query2 = torch.tensor([0.0, 1.0, 0.0])
weights2 = torch.softmax(vectors @ query2, dim=0)
for word, w in zip(words, weights2):
    print(f"{word:6s} {w.item():.2f} {'#' * int(w.item() * 40)}")

print("\n=== 5. Many queries at once (a matrix) ===")
queries = torch.stack([query, query2])        # (2, 3)
all_scores = queries @ vectors.T              # (2, 4): every query vs every word
all_weights = torch.softmax(all_scores, dim=1)
print("weights shape:", tuple(all_weights.shape), "= (queries, words)")
print(all_weights.round(decimals=2))

print("\n=== 6. Bigger scores make attention sharper ===")
for scale in [1, 5, 20]:
    w = torch.softmax(scores * scale, dim=0)
    print(f"scores x {scale:2d} -> {w.round(decimals=2).tolist()}")
