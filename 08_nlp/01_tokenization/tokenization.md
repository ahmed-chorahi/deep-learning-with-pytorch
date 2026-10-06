# 💬 Tokenization: Turning Text into Numbers

> 📚 **NLP (Natural Language Processing)** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🟢 Beginner
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Learn how text becomes numbers that a neural network can use.

## 📌 What is it?
**Tokenization** splits text into pieces (tokens, usually words). A **vocabulary** gives every token an id number. After that, sentences become lists of ids, and lists of ids become tensors.

## 🧠 Real-Life Analogy
Think of a phone book. Every person (word) gets a unique page number (id). To talk about people, the computer only uses page numbers.

---

## 💻 Example

Code from `tokenization.py` (run it with `python tokenization.py`):

```python
# Lesson 1: Tokenization
# Computers cannot read words, only numbers. So we turn text into numbers:
# text -> tokens (words) -> ids (numbers) -> tensor

import string
from collections import Counter

import torch

sentences = [
    "I love PyTorch!",
    "PyTorch is fun.",
    "I do not like bugs.",
    "Bugs are not fun!",
]

print("=== 1. Cleaning text ===")


def clean(text):
    text = text.lower()                                              # lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove . ! ?
    return text


for s in sentences:
    print(f"{s!r:25s} -> {clean(s)!r}")

print("\n=== 2. Tokens (split into words) ===")
all_tokens = [clean(s).split() for s in sentences]
for tokens in all_tokens:
    print(tokens)

print("\n=== 3. Counting words ===")
counts = Counter(word for tokens in all_tokens for word in tokens)
print(counts.most_common(5))

print("\n=== 4. Building a vocabulary (word -> number) ===")
word_to_id = {"<pad>": 0, "<unk>": 1}        # special tokens: padding and unknown
for word in counts:
    word_to_id[word] = len(word_to_id)
id_to_word = {i: w for w, i in word_to_id.items()}
print("vocabulary size:", len(word_to_id))
print(word_to_id)


def encode(text):
    return [word_to_id.get(word, word_to_id["<unk>"]) for word in clean(text).split()]


def decode(ids):
    return " ".join(id_to_word[i] for i in ids)


print("\n=== 5. Encoding and decoding ===")
ids = encode("I love PyTorch!")
print("encoded:", ids)
print("decoded:", decode(ids))

print("\n=== 6. Unknown words ===")
print("encode('I love tensors') ->", encode("I love tensors"))
print("decoded back            ->", decode(encode("I love tensors")))

print("\n=== 7. Padding (all sentences must have the same length) ===")
encoded = [encode(s) for s in sentences]
max_len = max(len(e) for e in encoded)
padded = [e + [word_to_id["<pad>"]] * (max_len - len(e)) for e in encoded]
batch = torch.tensor(padded)
print(batch)
print("batch shape:", tuple(batch.shape), "= (sentences, max_len)")

print("\n=== 8. Mask: which positions are real words? ===")
mask = batch != word_to_id["<pad>"]
print(mask)
print("real words per sentence:", mask.sum(dim=1).tolist())
```

## 📤 Expected Output

```text
=== 1. Cleaning text ===
'I love PyTorch!'         -> 'i love pytorch'
'PyTorch is fun.'         -> 'pytorch is fun'
'I do not like bugs.'     -> 'i do not like bugs'
'Bugs are not fun!'       -> 'bugs are not fun'

=== 2. Tokens (split into words) ===
['i', 'love', 'pytorch']
['pytorch', 'is', 'fun']
['i', 'do', 'not', 'like', 'bugs']
['bugs', 'are', 'not', 'fun']

=== 3. Counting words ===
[('i', 2), ('pytorch', 2), ('fun', 2), ('not', 2), ('bugs', 2)]

=== 4. Building a vocabulary (word -> number) ===
vocabulary size: 12
{'<pad>': 0, '<unk>': 1, 'i': 2, 'love': 3, 'pytorch': 4, 'is': 5, 'fun': 6, 'do': 7, 'not': 8, 'like': 9, 'bugs': 10, 'are': 11}

=== 5. Encoding and decoding ===
encoded: [2, 3, 4]
decoded: i love pytorch

=== 6. Unknown words ===
encode('I love tensors') -> [2, 3, 1]
decoded back            -> i love <unk>

=== 7. Padding (all sentences must have the same length) ===
tensor([[ 2,  3,  4,  0,  0],
        [ 4,  5,  6,  0,  0],
        [ 2,  7,  8,  9, 10],
        [10, 11,  8,  6,  0]])
batch shape: (4, 5) = (sentences, max_len)

=== 8. Mask: which positions are real words? ===
tensor([[ True,  True,  True, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True,  True,  True],
        [ True,  True,  True,  True, False]])
real words per sentence: [3, 3, 5, 4]
```

## 🔍 Explanation
1. Part 1 lowercases the text and removes punctuation with `str.translate`.
2. Part 2 splits each sentence into words.
3. Part 3 counts words with `Counter`.
4. Part 4 builds the vocabulary; `<pad>` = 0 and `<unk>` = 1 are special tokens.
5. Part 5 encodes text into ids and decodes ids back into text.
6. Part 6 shows that unknown words become `<unk>`.
7. Part 7 pads all sentences to the same length and makes a tensor.
8. Part 8 builds a mask that marks the real words.

---

## 📐 The Text Pipeline

```
"I love PyTorch!"
      │ clean
      ▼
"i love pytorch"
      │ split
      ▼
["i", "love", "pytorch"]
      │ vocabulary lookup
      ▼
[2, 3, 4]
      │ pad + torch.tensor
      ▼
tensor of shape (sentences, max_len)
```

| Special token | Id | Meaning |
|---------------|----|---------|
| `<pad>` | 0 | filler so all sentences have the same length |
| `<unk>` | 1 | a word we have never seen |

## 📝 Important Points

- ✅ Neural networks only understand numbers.
- ✅ The vocabulary is built from the training data only.
- ✅ Padding makes sentences the same length so they fit in one tensor.
- ✅ A mask tells the model which positions are real words.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Building the vocabulary from test data | Use training data only |
| Forgetting an `<unk>` token | New words would crash the lookup |
| Different lengths in one tensor | Pad to the same length |

## 🧪 Try It Yourself

- [ ] Add a new sentence and print its encoding.
- [ ] Encode a sentence with two unknown words.
- [ ] Pad on the left instead of the right.

## ❓ Quick Questions

1. Why do we need a vocabulary?
2. What is the `<pad>` token for?
3. What happens to a word that is not in the vocabulary?

---

## 🏁 Summary

> 💡 Text → tokens → ids → padded tensor is the first step of every NLP project.

➡️ **Next:** [Embeddings](../02_embeddings/embeddings.md)
