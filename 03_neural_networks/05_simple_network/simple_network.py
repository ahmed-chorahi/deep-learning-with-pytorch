# Lesson 5: A simple neural network
# Linear -> ReLU -> Linear. It takes 4 numbers and picks one of 3 classes.

import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)      # input  -> hidden
        x = self.relu(x)     # non-linearity
        x = self.fc2(x)      # hidden -> output scores
        return x


model = SimpleNetwork(input_size=4, hidden_size=8, output_size=3)
print("=== 1. The model ===")
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

print("\n=== 2. Forward pass with random data ===")
x = torch.rand(5, 4)                       # 5 samples, 4 features
scores = model(x)
print("output shape:", tuple(scores.shape))
print("predicted classes:", scores.argmax(dim=1))

print("\n=== 3. Loss for random weights ===")
labels = torch.tensor([0, 1, 2, 1, 0])
loss_fn = nn.CrossEntropyLoss()
print("loss:", round(loss_fn(scores, labels).item(), 4))

print("\n=== 4. Teaching it on these 5 samples ===")
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for epoch in range(60):
    loss = loss_fn(model(x), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 15 == 0:
        print(f"epoch {epoch:2d} | loss = {loss.item():.4f}")

print("\n=== 5. After training ===")
predictions = model(x).argmax(dim=1)
print("true labels     :", labels.tolist())
print("predicted labels:", predictions.tolist())
accuracy = (predictions == labels).float().mean().item() * 100
print(f"accuracy on these samples: {accuracy:.0f}%")
print("(it memorized 5 samples; real projects use separate test data)")
