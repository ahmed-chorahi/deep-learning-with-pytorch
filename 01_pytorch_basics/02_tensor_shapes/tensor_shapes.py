# Lesson 2: Tensor shapes
# Most PyTorch errors are shape errors, so learn to change shapes confidently.

import torch


def show(name, t):
    # small helper so every print looks the same
    print(f"{name:22s} shape = {tuple(t.shape)}")


x = torch.arange(12)
show("original", x)

print("\n=== reshape ===")
show("reshape(3, 4)", x.reshape(3, 4))
show("reshape(2, 6)", x.reshape(2, 6))
show("reshape(2, -1)", x.reshape(2, -1))     # -1 = "you work it out"
show("reshape(2, 3, 2)", x.reshape(2, 3, 2))

# Wrong reshape: 12 numbers cannot fit into 5 x 3 = 15 places
try:
    x.reshape(5, 3)
except RuntimeError as error:
    print("Error caught:", str(error)[:60], "...")

print("\n=== squeeze and unsqueeze ===")
v = torch.tensor([1, 2, 3])
show("v", v)
show("v.unsqueeze(0)", v.unsqueeze(0))       # (1, 3)  one row
show("v.unsqueeze(1)", v.unsqueeze(1))       # (3, 1)  one column
show("zeros(1,3,1).squeeze()", torch.zeros(1, 3, 1).squeeze())

print("\n=== transpose and permute ===")
m = torch.rand(2, 3)
show("m", m)
show("m.T", m.T)
image = torch.rand(28, 28, 3)                # height, width, channels
show("image (H, W, C)", image)
show("permute(2, 0, 1)", image.permute(2, 0, 1))   # PyTorch wants (C, H, W)

print("\n=== flatten ===")
show("flatten of (2,3,4)", torch.rand(2, 3, 4).flatten())

print("\n=== batches ===")
one_image = torch.rand(1, 28, 28)
batch = one_image.unsqueeze(0)               # models expect a batch dimension
show("one image", one_image)
show("batch of 1 image", batch)
show("batch of 64 images", torch.rand(64, 1, 28, 28))
