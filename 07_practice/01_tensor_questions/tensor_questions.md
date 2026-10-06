# 🧪 Tensor Questions

> 📚 **Practice** &nbsp;|&nbsp; 🧩 Lesson 1 of 5 &nbsp;|&nbsp; ⏱️ ~20 min &nbsp;|&nbsp; 🎓 🔵 Practice
>
> Progress: `▰▱▱▱▱` 1/5

---

## 🎯 Learning Goal
Test your knowledge of tensors with 8 questions.

## 📌 What is it?
Each question is answered in the code. Try to answer before you run it.

## 🧠 Real-Life Analogy
A quick quiz after the lesson: it shows what you've remembered.

---

## 💻 Example

Code from `tensor_questions.py` (run it with `python tensor_questions.py`):

```python
# Tensor questions - try to answer each one, then compare with the code.
import torch

score = 0
total = 0


def check(question, ok):
    global score, total
    total += 1
    score += ok
    print(f"Q{question}:", "correct" if ok else "wrong")


# Q1: How do you create a 3x3 tensor filled with zeros?
q1 = torch.zeros(3, 3)
print("Q1:\n", q1)
check(1, q1.shape == (3, 3))

# Q2: What is the shape of torch.rand(5, 2, 4)?
q2 = torch.rand(5, 2, 4).shape
print("Q2:", q2)
check(2, q2 == (5, 2, 4))

# Q3: Turn [1, 2, 3, 4, 5, 6] into shape (3, 2)
q3 = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(3, 2)
print("Q3:\n", q3)
check(3, q3.shape == (3, 2))

# Q4: What is the difference between * and @ ?
a = torch.tensor([[1, 2], [3, 4]])
print("Q4: * is element-wise:\n", a * a, "\n    @ is matrix multiplication:\n", a @ a)
check(4, (a @ a)[0, 0].item() == 7 and (a * a)[0, 0].item() == 1)

# Q5: Largest value and its position in [3, 9, 2, 7]
v = torch.tensor([3, 9, 2, 7])
print("Q5:", v.max().item(), "at index", v.argmax().item())
check(5, v.argmax().item() == 1)

# Q6: Replace negative numbers with 0
t = torch.tensor([-1, 2, -3, 4])
t[t < 0] = 0
print("Q6:", t)
check(6, t.tolist() == [0, 2, 0, 4])

# Q7: Add a batch dimension to a (28, 28) tensor
q7 = torch.rand(28, 28).unsqueeze(0).shape
print("Q7:", q7)
check(7, q7 == (1, 28, 28))

# Q8: Average of each row of a (3, 4) tensor
q8 = torch.rand(3, 4).mean(dim=1).shape
print("Q8:", q8)
check(8, q8 == (3,))

# Q9: How many elements are in a tensor of shape (4, 5, 6)?
q9 = torch.zeros(4, 5, 6).numel()
print("Q9:", q9)
check(9, q9 == 120)

print(f"\nScore: {score}/{total}")
```

## 📤 Expected Output

```text
Q1:
 tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
Q1: correct
Q2: torch.Size([5, 2, 4])
Q2: correct
Q3:
 tensor([[1, 2],
        [3, 4],
        [5, 6]])
Q3: correct
Q4: * is element-wise:
 tensor([[ 1,  4],
        [ 9, 16]]) 
    @ is matrix multiplication:
 tensor([[ 7, 10],
        [15, 22]])
Q4: correct
Q5: 9 at index 1
Q5: correct
Q6: tensor([0, 2, 0, 4])
Q6: correct
Q7: torch.Size([1, 28, 28])
Q7: correct
Q8: torch.Size([3])
Q8: correct
Q9: 120
Q9: correct

Score: 9/9

Note: This lesson uses random numbers, so your values will be different.
```

## 🔍 Explanation
1. A `check()` function keeps a score.
2. Q1-Q3: create, check shape and reshape.
3. Q4: `*` vs `@`.
4. Q5-Q6: `max`, `argmax` and masks.
5. Q7-Q8: batch dimension and row averages.
6. Q9: counting elements with `numel()`.
7. The score is printed at the end.

---

## 📐 Shape Answers

```
Q2: (5, 2, 4)
Q3: (3, 2)
Q7: (1, 28, 28)
Q8: (3,)
```

## 📝 Important Points

- ✅ Try first, then peek at the answer.
- ✅ Print shapes while you think.
- ✅ Repeat the questions after a few days.

## ⚠️ Common Mistakes

| ❌ Common mistake | ✅ What to do |
|---|---|
| Skipping the thinking part | Predict the answer before running |

## 🧪 Try It Yourself

- [ ] Write 3 questions of your own.
- [ ] Change Q6 to replace values above 3 with 3.
- [ ] Explain Q4 in your own words.

## ❓ Quick Questions

1. What is `torch.zeros(3, 3)`?
2. What does `argmax` return?
3. What does `unsqueeze(0)` do to a (28, 28) tensor?

---

## 🏁 Summary

> 💡 Nine tensor questions with an automatic score.

➡️ **Next:** [Neural Network Questions](../02_neural_network_questions/neural_network_questions.md)
