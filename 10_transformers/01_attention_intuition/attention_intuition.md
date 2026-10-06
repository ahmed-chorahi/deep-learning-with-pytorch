# 🤖 Attention Intuition: Where Should I Look?

> 📚 **Transformers** &nbsp;|&nbsp; 🧩 Lesson 1 of 10 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▱▱▱▱▱▱▱▱▱` 1/10

---

## 🎯 Learning Goal
Understand the idea of attention (compare, weigh, mix) before any Transformer maths.

## 📌 What is it?
**Attention** lets a word look at other words and decide how much each one matters. It does three things: (1) compare a *query* with every word using a dot product, (2) turn the scores into weights with **softmax**, (3) build the output as a weighted mix of the word vectors.

## 🧠 Real-Life Analogy
Imagine a library. Your question (the query) is compared with the title on each book (the keys). The books that match best get the most of your time, and what you take away (the output) is mostly from those books.

## 📖 Key Terms

| 📖 Term | Meaning |
|---|---|
| **Query** | what the current word is looking for |
| **Key** | what each word offers to be matched against |
| **Value** | the information that is actually passed on |
| **Score** | how well a query matches a key (dot product) |
| **Weight** | a score after softmax; all weights add up to 1 |

---

## 💻 Example

Code from `attention_intuition.py` (run it with `python attention_intuition.py`):

```python
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
```

## 📤 Expected Output

```text
=== 1. A query: 'I am looking for animals' ===
scores : {'cat': 0.8999999761581421, 'dog': 0.800000011920929, 'car': 0.10000000149011612, 'apple': 0.0}

=== 2. Softmax turns scores into weights that add up to 1 ===
cat    0.36 ##############
dog    0.33 #############
car    0.16 ######
apple  0.15 #####
sum of weights: 1.0

=== 3. The output is a weighted mix of the word vectors ===
output: tensor([0.6000, 0.2600, 0.1300])
-> mostly 'animal', because cat and dog got the biggest weights

=== 4. A different query: 'I am looking for vehicles' ===
cat    0.19 #######
dog    0.21 ########
car    0.42 ################
apple  0.19 #######

=== 5. Many queries at once (a matrix) ===
weights shape: (2, 4) = (queries, words)
tensor([[0.3600, 0.3300, 0.1600, 0.1500],
        [0.1900, 0.2100, 0.4200, 0.1900]])

=== 6. Bigger scores make attention sharper ===
scores x  1 -> [0.36000001430511475, 0.33000001311302185, 0.1599999964237213, 0.15000000596046448]
scores x  5 -> [0.6100000143051147, 0.3700000047683716, 0.009999999776482582, 0.009999999776482582]
scores x 20 -> [0.8799999952316284, 0.11999999731779099, 0.0, 0.0]
```

## 🔍 Explanation
1. We give four words tiny hand-made vectors: `[animal, vehicle, food]`.
2. Part 1: the query `[1, 0, 0]` means 'I am looking for animals'. Dot products give high scores for cat and dog.
3. Part 2: softmax turns the scores into weights that add up to 1, drawn as text bars.
4. Part 3: the output is `weights @ vectors`, a mix that is mostly 'animal'.
5. Part 4: a vehicle query moves the attention to the word car.
6. Part 5: many queries at once become one matrix multiplication, `queries @ vectors.T`.
7. Part 6: multiplying the scores makes the weights sharper, almost 'all or nothing'.

---

## 📐 Attention in Three Steps

```
 query ──┐
         ├─► scores = query · each key ──► softmax ──► weights ──┐
 keys ───┘                                                        ├─► weighted sum ─► output
 values ────────────────────────────────────────────────────────┘
```

| Step | Maths | Shape in this lesson |
|------|-------|----------------------|
| Compare | `scores = vectors @ query` | `(4,)` |
| Weigh | `weights = softmax(scores)` | `(4,)` |
| Mix | `output = weights @ vectors` | `(3,)` |
| Many queries | `queries @ vectors.T` | `(2, 4)` |

## 📝 Important Points

- ✅ Attention is a *weighted average*, nothing magical.
- ✅ Softmax makes the weights positive and add up to 1.
- ✅ Bigger scores give sharper attention.
- ✅ The same idea is used inside every Transformer.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Thinking attention picks only one word | It mixes all words; some just get tiny weights |
| Applying softmax over the wrong dimension | Softmax goes over the words being looked at |

## 🧪 Try It Yourself

- [ ] Make a query that likes both animals and food.
- [ ] Add a fifth word 'bus' with a vehicle-like vector.
- [ ] Print the weights for `scores * 0.1` (very flat attention).

## ❓ Quick Questions

1. What are the three steps of attention?
2. Why do we use softmax?
3. What happens to the weights when the scores are multiplied by 20?

---

## 🏁 Summary

> 💡 Attention = compare, softmax, weighted mix.

➡️ **Next:** [Scaled Dot Product Attention](../02_scaled_dot_product_attention/scaled_dot_product_attention.md)
