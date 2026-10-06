# 🎓 AI Undergrad PyTorch Journey

> My step-by-step path from "What is a tensor?" to a tiny GPT: **59 small lessons**, each with runnable code and a friendly explanation.

Made for undergraduate students who are starting with AI and want to **learn by running real code**.

---

## 🗺️ The Learning Path

| | Folder | Topic | Lessons | Time | Data |
|---|--------|-------|---------|------|------|
| 🔢 | [01_pytorch_basics](01_pytorch_basics/README.md) | PyTorch Basics | 5 | ~1 h 25 min | ✅ offline |
| 📈 | [02_autograd](02_autograd/README.md) | Autograd | 5 | ~1 h 30 min | ✅ offline |
| 🧠 | [03_neural_networks](03_neural_networks/README.md) | Neural Networks | 5 | ~1 h 45 min | ✅ offline |
| 🏋️ | [04_training](04_training/README.md) | Training | 5 | ~1 h 40 min | ✅ offline |
| 👁️ | [05_computer_vision](05_computer_vision/README.md) | Computer Vision | 5 | ~1 h 55 min | 🌐 once |
| 🚀 | [06_projects](06_projects/README.md) | Projects | 9 | ~2 h 50 min | 🌐 once |
| 🧪 | [07_practice](07_practice/README.md) | Practice | 5 | ~2 h 5 min | ✅ offline |
| 💬 | [08_nlp](08_nlp/README.md) | NLP | 5 | ~2 h 0 min | ✅ offline |
| 🖼️ | [09_cnn](09_cnn/README.md) | CNN | 5 | ~2 h 5 min | ✅ offline |
| 🤖 | [10_transformers](10_transformers/README.md) | Transformers | 10 | ~5 h 25 min | ✅ offline |

Follow the folders in order. Each one builds on the one before.

```
01 Basics ─► 02 Autograd ─► 03 Neural Nets ─► 04 Training ─► 05 Vision ─► 06 Projects
                                                    │              │
                                                    ▼              ▼
                                               08 NLP ───────► 10 Transformers
                                                               09 CNN (deeper look)
                                              07 Practice (test yourself any time)
```

---

## 📦 How a lesson is organized

Every lesson is **one folder with two files**:

```
01_tensors/
├── tensors.py     💻 code you run (small, with simple comments)
└── tensors.md     📖 the lesson (goal, analogy, expected output, explanation,
                      common mistakes, exercises and quick questions)
```

```
 One complete lesson
 ├── Read the .md  → understand the idea
 └── Run the .py   → see it work, then change it and experiment
```

---

## 🚀 Quick Start

**1. Get the code** and open a terminal in the main folder.

**2. (Optional) make a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**3. Install the libraries**

```bash
pip install -r requirements.txt
```

**4. Run your first lesson**

```bash
cd 01_pytorch_basics/01_tensors
python tensors.py
```

Then open `tensors.md` next to it and compare the output with the **Expected Output** section.

---

## 🧭 How to study

1. 📖 Read the lesson `.md` (start with **Learning Goal** and **Real-Life Analogy**).
2. 💻 Run the `.py` file and compare with the expected output.
3. 🔧 Change a number or a layer and run again. Breaking things is how you learn.
4. 🧪 Do **Try It Yourself** and answer the **Quick Questions**.
5. ✅ Tick the lesson in [PROGRESS.md](PROGRESS.md).

**Suggested schedule:** one folder per week gives you the whole path in about 10 weeks. Each lesson takes 15 to 40 minutes.

---

## 🧰 Extras

| File | What it is |
|------|-----------|
| [CHEATSHEET.md](CHEATSHEET.md) | one page with the most important PyTorch code and common errors |
| [PROGRESS.md](PROGRESS.md) | a checklist of all 59 lessons |
| [requirements.txt](requirements.txt) | libraries to install |
| `README.md` in each folder | a table of that section's lessons |

---

## 📝 Good to know

- **Python 3.9+** and **PyTorch 2.0+** are recommended. A GPU is *not* needed; everything runs on a normal CPU.
- Sections 05 and 06 download MNIST / Fashion-MNIST the first time you run them. Each lesson or project folder keeps its own `data` folder (a few dozen MB each), so delete those `data` folders when you finish. Everything else works offline.
- The code uses only `torch`, `torchvision` and `numpy`. Text and image examples are small and made by hand, so they run fast.
- Numbers in the **Expected Output** blocks can differ a little on your computer because of random numbers. The lessons for MNIST and Fashion-MNIST show approximate values like `98.xx%`.
- Small datasets give very high accuracy. That is good for learning, but real projects need much more data and a separate test set.

---

## 🆘 Something went wrong?

| Problem | Try this |
|---------|----------|
| `ModuleNotFoundError: torch` | run `pip install -r requirements.txt` |
| `ModuleNotFoundError: model` in a project | run the script from its own folder, e.g. `cd 02_train` then `python train.py` |
| `FileNotFoundError: model.pth` | run the project's `02_train/train.py` before `03_predict/predict.py` |
| MNIST will not download | check your internet connection, then run again |
| Different numbers than the lesson | normal; random numbers change between runs |
| An error you do not understand | look at the table at the end of [CHEATSHEET.md](CHEATSHEET.md) |

---

Happy learning! 🎉 Start here: [01_pytorch_basics](01_pytorch_basics/README.md)
