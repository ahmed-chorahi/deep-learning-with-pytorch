# Final practice: put everything together
# Goal: train a small network to classify points into 3 groups.
# Try changing the numbers marked "TRY" and see what happens!
import torch
import torch.nn as nn

torch.manual_seed(1)

# --- 1. Make fake data: 3 groups of points around different centers ---
centers = torch.tensor([[0.0, 0.0], [3.0, 3.0], [0.0, 4.0]])
x = torch.cat([center + torch.randn(100, 2) for center in centers])
y = torch.cat([torch.full((100,), i) for i in range(3)])

# --- 2. Shuffle and split into train / test ---
order = torch.randperm(300)
x, y = x[order], y[order]
x_train, y_train = x[:240], y[:240]
x_test, y_test = x[240:], y[240:]
print("train:", tuple(x_train.shape), "| test:", tuple(x_test.shape))

# --- 3. Build the model ---
HIDDEN = 16                                   # TRY: 4, 32
model = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 3))

# --- 4. Loss and optimizer ---
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)    # TRY: 0.1, 0.001

# --- 5. Train ---
for epoch in range(100):                      # TRY: 20, 300
    model.train()
    loss = loss_fn(model(x_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | loss = {loss.item():.4f}")

# --- 6. Evaluate ---
model.eval()
with torch.no_grad():
    predictions = model(x_test).argmax(dim=1)
accuracy = (predictions == y_test).float().mean().item()
print(f"Test accuracy: {accuracy * 100:.1f}%")

# --- 7. Confusion table (rows = true group, columns = predicted group) ---
print("\nConfusion table:")
print("        pred0 pred1 pred2")
for true_group in range(3):
    row = [((predictions == p) & (y_test == true_group)).sum().item() for p in range(3)]
    print(f"true {true_group}  {row[0]:5d} {row[1]:5d} {row[2]:5d}")

# --- 8. Predict a new point ---
new_point = torch.tensor([[2.8, 3.2]])
with torch.no_grad():
    probs = torch.softmax(model(new_point), dim=1)[0]
print("\nNew point [2.8, 3.2] -> group", probs.argmax().item(), "(expected 1)")
print("probabilities:", [round(p, 2) for p in probs.tolist()])
