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
