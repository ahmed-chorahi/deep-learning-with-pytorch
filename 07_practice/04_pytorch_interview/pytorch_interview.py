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
