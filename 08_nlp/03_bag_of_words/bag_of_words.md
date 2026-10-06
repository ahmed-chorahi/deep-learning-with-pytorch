# 💬 Bag of Words: A Simple Sentiment Classifier

> 📚 **NLP (Natural Language Processing)** &nbsp;|&nbsp; 🧩 Lesson 3 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🟡 Intermediate
>
> Progress: `▰▰▰▱▱` 3/5

---

## 🎯 Learning Goal
Build a first text classifier by counting words.

## 📌 What is it?
In a **bag of words**, a sentence is represented by how many times each vocabulary word appears. Word order is ignored, like shaking all the words of a sentence in a bag.

## 🧠 Real-Life Analogy
It's like judging a shopping basket by listing what is inside, without caring in which order the items were put in.

---

## 💻 Example

Code from `bag_of_words.py` (run it with `python bag_of_words.py`):

```python
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
```

## 📤 Expected Output

```text
vocabulary size: 31

=== 1. A bag-of-words vector ===
i love this movie -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
words present: ['i', 'love', 'movie', 'this']

train tensor shape: (20, 31) = (sentences, vocabulary)

=== 2. Training ===
epoch  25 | loss = 0.0494
epoch  50 | loss = 0.0192
epoch  75 | loss = 0.0126
epoch 100 | loss = 0.0093

=== 3. Accuracy ===
train accuracy: 100%
test accuracy : 100%

=== 4. Predicting new sentences ===
'what a great movie'         -> positive (97%)
'i hate this boring film'    -> negative (98%)
'so happy'                   -> positive (94%)

=== 5. Which words matter most? ===
most negative words: ['hate', 'sad', 'awful', 'an']
most positive words: ['wonderful', 'lovely', 'love', 'happy']
```

## 🔍 Explanation
1. A small hand-made dataset has 20 training and 4 test sentences (positive = 1, negative = 0).
2. A vocabulary maps each word to a position.
3. `bag_of_words()` makes a vector of counts; unknown words are ignored.
4. Part 2 trains `nn.Linear(vocab, 2)` with Adam and CrossEntropyLoss.
5. Part 3 prints train and test accuracy.
6. Part 4 predicts brand-new sentences with probabilities.
7. Part 5 looks at the weights to see which words push towards positive or negative.

---

## 📐 How the Model Sees a Sentence

```
vocabulary: [awful, bad, good, great, happy, hate, love, ...]

"i love this movie"  ->  [0, 0, 0, 0, 0, 0, 1, ...]   (a vector of counts)
                              │ nn.Linear(vocab_size, 2)
                              ▼
                       [score_negative, score_positive]
```

Shapes: `x_train` is `(sentences, vocabulary)`, the model output is `(sentences, 2)`.

## 📝 Important Points

- ✅ Simple models can work well on simple text.
- ✅ Word order is lost, so 'not good' looks like 'good' + 'not'.
- ✅ The weights show which words matter most.
- ✅ Always evaluate on sentences the model has not seen.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Counting words from the test set in the vocabulary | Build it from training data |
| Believing 100% accuracy on 4 sentences means perfect | The test set is tiny; use more data in real projects |

## 🧪 Try It Yourself

- [ ] Add 5 new training sentences.
- [ ] Predict a sentence with the word 'not'. What happens?
- [ ] Print the vocabulary size.

## ❓ Quick Questions

1. What is a bag of words?
2. What information does it lose?
3. Why do we check the weights at the end?

---

## 🏁 Summary

> 💡 Counting words and a linear layer is already a working sentiment classifier.

⬅️ **Previous:** [Embeddings](../02_embeddings/embeddings.md)  |  ➡️ **Next:** [Rnn Basics](../04_rnn_basics/rnn_basics.md)
