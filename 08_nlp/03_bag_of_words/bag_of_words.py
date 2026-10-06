# Lesson 3: Bag of words sentiment classifier
# Count how many times each word appears, then use a linear layer to decide
# if a sentence is positive (1) or negative (0). Word order is ignored.

import string
import torch
import torch.nn as nn

torch.manual_seed(0)

train_data = [
    ("i love this movie", 1), ("what a great film", 1), ("good acting and nice story", 1),
    ("i am so happy", 1), ("great fun and good music", 1), ("nice and lovely", 1),
    ("love the happy ending", 1), ("a good great movie", 1), ("so nice i love it", 1),
    ("wonderful and happy", 1),
    ("i hate this movie", 0), ("what an awful film", 0), ("bad acting and boring story", 0),
    ("i am so sad", 0), ("awful and bad music", 0), ("boring and ugly", 0),
    ("hate the sad ending", 0), ("a bad awful movie", 0), ("so boring i hate it", 0),
    ("terrible and sad", 0),
]
test_data = [
    ("what a lovely happy film", 1), ("great music and nice acting", 1),
    ("a sad and boring movie", 0), ("i hate the awful story", 0),
]

# ----- vocabulary -----
words = sorted({w for sentence, _ in train_data for w in sentence.split()})
word_to_id = {w: i for i, w in enumerate(words)}
print("vocabulary size:", len(words))


def bag_of_words(sentence):
    vector = torch.zeros(len(words))
    for w in sentence.split():
        if w in word_to_id:                     # unknown words are ignored
            vector[word_to_id[w]] += 1
    return vector


print("\n=== 1. A bag-of-words vector ===")
example = "i love this movie"
vec = bag_of_words(example)
print(example, "->", vec.int().tolist())
print("words present:", [words[i] for i in vec.nonzero().flatten().tolist()])


def to_tensors(data):
    x = torch.stack([bag_of_words(s) for s, _ in data])
    y = torch.tensor([label for _, label in data])
    return x, y


x_train, y_train = to_tensors(train_data)
x_test, y_test = to_tensors(test_data)
print("\ntrain tensor shape:", tuple(x_train.shape), "= (sentences, vocabulary)")

# ----- model -----
model = nn.Linear(len(words), 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print("\n=== 2. Training ===")
for epoch in range(1, 101):
    loss = loss_fn(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d} | loss = {loss.item():.4f}")


def accuracy(x, y):
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item() * 100


print("\n=== 3. Accuracy ===")
print(f"train accuracy: {accuracy(x_train, y_train):.0f}%")
print(f"test accuracy : {accuracy(x_test, y_test):.0f}%")

print("\n=== 4. Predicting new sentences ===")
for sentence in ["what a great movie", "i hate this boring film", "so happy"]:
    with torch.no_grad():
        probs = torch.softmax(model(bag_of_words(sentence).unsqueeze(0)), dim=1)[0]
    label = "positive" if probs.argmax().item() == 1 else "negative"
    print(f"{sentence!r:28s} -> {label} ({probs.max().item() * 100:.0f}%)")

print("\n=== 5. Which words matter most? ===")
# weight difference = how much a word pushes the answer towards "positive"
push = (model.weight[1] - model.weight[0]).detach()
order = push.argsort()
print("most negative words:", [words[i] for i in order[:4].tolist()])
print("most positive words:", [words[i] for i in order[-4:].tolist()])
