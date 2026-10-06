# 🧪 Pytorch Interview

> 📚 **Practice** &nbsp;|&nbsp; 🧩 Lesson 4 of 5 &nbsp;|&nbsp; ⏱️ ~25 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▰▰▰▱` 4/5

---

## 🎯 Learning Goal
Prepare for common beginner PyTorch interview questions.

## 📌 What is it?
Short, clear answers to the most common beginner questions, with small demos where possible.

## 🧠 Real-Life Analogy
Like a cheat sheet you read the night before an interview.

---

## 💻 Example

Code from `pytorch_interview.py` (run it with `python pytorch_interview.py`):

```python
# Common beginner PyTorch interview questions with short answers and small demos
import torch
import torch.nn as nn

questions = [
    ("What is a tensor?",
     "A multi-dimensional array that can run on a GPU and track gradients."),
    ("What does requires_grad=True do?",
     "PyTorch records operations on the tensor so it can compute gradients later."),
    ("Why call optimizer.zero_grad()?",
     "Gradients accumulate by default, so we clear them at every step."),
    ("model.train() vs model.eval()?",
     "They switch layers like Dropout and BatchNorm between training and testing behaviour."),
    ("Why use torch.no_grad() when predicting?",
     "It skips gradient tracking, saving memory and time."),
    ("What is an epoch? A batch?",
     "Epoch = one full pass over the data. Batch = a small group of samples processed together."),
    ("How do you save a model?",
     "torch.save(model.state_dict(), 'model.pth'), then load_state_dict(torch.load('model.pth'))."),
    ("What does CrossEntropyLoss expect?",
     "Raw scores (logits) of shape (batch, classes) and integer class labels."),
    ("Why do we need activation functions?",
     "Without them a stack of linear layers is just one linear layer."),
    ("What is overfitting?",
     "The model memorizes training data and performs badly on new data."),
]

for number, (q, a) in enumerate(questions, start=1):
    print(f"Q{number}: {q}")
    print(f"A{number}: {a}\n")

print("--- Small demos ---")

# Demo 1: gradient
x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward()
print("Demo 1: d(x^2)/dx at x=2 ->", x.grad.item())

# Demo 2: view vs reshape
t = torch.arange(6)
print("Demo 2: view ->", tuple(t.view(2, 3).shape), "| reshape ->", tuple(t.reshape(3, 2).shape))

# Demo 3: eval mode turns dropout off
drop = nn.Dropout(0.5)
data = torch.ones(8)
drop.train()
print("Demo 3: train mode ->", drop(data))
drop.eval()
print("        eval mode  ->", drop(data))

# Demo 4: gradients accumulate
w = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    (w * 2).backward()
print("Demo 4: gradient after 3 backward calls ->", w.grad.item(), "(not 2!)")
```

## 📤 Expected Output

```text
Q1: What is a tensor?
A1: A multi-dimensional array that can run on a GPU and track gradients.

Q2: What does requires_grad=True do?
A2: PyTorch records operations on the tensor so it can compute gradients later.

Q3: Why call optimizer.zero_grad()?
A3: Gradients accumulate by default, so we clear them at every step.

Q4: model.train() vs model.eval()?
A4: They switch layers like Dropout and BatchNorm between training and testing behaviour.

Q5: Why use torch.no_grad() when predicting?
A5: It skips gradient tracking, saving memory and time.

Q6: What is an epoch? A batch?
A6: Epoch = one full pass over the data. Batch = a small group of samples processed together.

Q7: How do you save a model?
A7: torch.save(model.state_dict(), 'model.pth'), then load_state_dict(torch.load('model.pth')).

Q8: What does CrossEntropyLoss expect?
A8: Raw scores (logits) of shape (batch, classes) and integer class labels.

Q9: Why do we need activation functions?
A9: Without them a stack of linear layers is just one linear layer.

Q10: What is overfitting?
A10: The model memorizes training data and performs badly on new data.

--- Small demos ---
Demo 1: d(x^2)/dx at x=2 -> 4.0
Demo 2: view -> (2, 3) | reshape -> (3, 2)
Demo 3: train mode -> tensor([2., 2., 0., 2., 0., 2., 0., 2.])
        eval mode  -> tensor([1., 1., 1., 1., 1., 1., 1., 1.])
Demo 4: gradient after 3 backward calls -> 6.0 (not 2!)
```

## 🔍 Explanation
1. A list of 10 question/answer pairs is printed with a loop.
2. Demo 1: gradient of x².
3. Demo 2: `view` vs `reshape`.
4. Demo 3: dropout in train vs eval mode.
5. Demo 4: gradients accumulate.

---

## 📐 Key Answers at a Glance

| Topic | One-line answer |
|-------|-----------------|
| tensor | multi-dimensional array with GPU + gradient support |
| `zero_grad` | gradients accumulate, so clear them each step |
| `eval()` | switches Dropout/BatchNorm to test behavior |
| `no_grad` | saves memory when predicting |

## 📝 Important Points

- ✅ Explain in your own words, not just memorized lines.
- ✅ Be ready to write a training loop from memory.
- ✅ Know the shapes in your examples.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Memorizing without understanding | Run the demos and explain in your own words |

## 🧪 Try It Yourself

- [ ] Answer each question out loud without looking.
- [ ] Add 3 more questions of your own.
- [ ] Write the training loop on paper.

## ❓ Quick Questions

1. Why call `optimizer.zero_grad()`?
2. What is the difference between an epoch and a batch?
3. What is the difference between `view` and `reshape`?

---

## 🏁 Summary

> 💡 Ten common questions plus four small demos.

⬅️ **Previous:** [Cnn Questions](../03_cnn_questions/cnn_questions.md)  |  ➡️ **Next:** [Final Practice](../05_final_practice/final_practice.md)
