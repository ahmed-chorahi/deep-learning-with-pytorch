# Load the trained model and predict test images (run ../02_train/train.py first)
import os
import sys
import torch
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_model"))
from model import Net

CLASS_NAMES = [str(i) for i in range(10)]

# ----- load the model -----
model = Net()
model.load_state_dict(torch.load(os.path.join(HERE, "..", "02_train", "model.pth")))
model.eval()

test_data = datasets.MNIST(root=os.path.join(HERE, "..", "data"), train=False,
                         download=True, transform=transforms.ToTensor())

# ----- predict the first 8 test images -----
correct = 0
for i in range(8):
    image, label = test_data[i]
    with torch.no_grad():
        probs = torch.softmax(model(image.unsqueeze(0)), dim=1)[0]
    predicted = probs.argmax().item()
    correct += predicted == label
    print(f"Image {i}: true = {CLASS_NAMES[label]:12s} | predicted = {CLASS_NAMES[predicted]:12s} "
          f"| confidence = {probs[predicted].item() * 100:5.1f}%")
print(f"{correct}/8 correct")

# ----- draw one image with text -----
image, label = test_data[0]
print("\nFirst test image (true label: %s):" % CLASS_NAMES[label])
for row in image[0][::2]:
    print("".join("#" if p > 0.5 else "." for p in row))
