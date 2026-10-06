# Neural network questions
import torch
import torch.nn as nn

# Q1: Build a network with 10 inputs, one hidden layer of 20 neurons (ReLU), 3 outputs
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
print("Q1:", model)

# Q2: How many parameters does nn.Linear(10, 20) have?  (10*20 weights + 20 biases)
count = sum(p.numel() for p in nn.Linear(10, 20).parameters())
print("Q2:", count, "= 10*20 + 20")

# Q3: What shape comes out for a batch of 8 samples?
print("Q3:", tuple(model(torch.rand(8, 10)).shape))

# Q4: Which loss for classification into 3 classes?  CrossEntropyLoss
logits = model(torch.rand(8, 10))
labels = torch.randint(0, 3, (8,))
print("Q4: loss =", round(nn.CrossEntropyLoss()(logits, labels).item(), 4))

# Q5: One training step
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss()(model(torch.rand(8, 10)), labels)   # forward + loss
optimizer.zero_grad()                                           # clear gradients
loss.backward()                                                 # backpropagate
optimizer.step()                                                # update weights
print("Q5: one training step done")

# Q6: Why do we need activation functions?
print("Q6: Without them, stacked linear layers collapse into one linear layer.")

# Q7: Count the parameters of the whole Q1 network
total = sum(p.numel() for p in model.parameters())
print("Q7:", total, "= 220 + 63")

# Q8: Train the network on a tiny made-up problem and watch the loss fall
torch.manual_seed(0)
x = torch.rand(30, 10)
y = (x.sum(dim=1) > 5).long()                  # class 1 if the numbers add up to > 5
net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.Adam(net.parameters(), lr=0.05)
for epoch in range(51):
    loss = nn.CrossEntropyLoss()(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 25 == 0:
        print(f"Q8: epoch {epoch:2d} loss = {loss.item():.3f}")
