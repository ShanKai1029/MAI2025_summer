# MAI HW2 - CLIP: Zero-Shot and Fine-Tuning

This repository contains the code for **Assignment 2**, focusing on the **CLIP** model. It includes experiments for **Zero-Shot** evaluation, **Linear Probing**, and **LoRA** fine-tuning.

- **Task1.ipynb**: Performs Zero-Shot classification on the **Flowers102** and **CUB-200-2011** datasets.
- **Task2.ipynb**: Performs fine-tuning (**Linear Probing** and **LoRA**) on the same datasets.

---

## 1. Setup

Before running the notebooks, you must set up the Conda environment and install the required packages.

```bash
# 1) Create the conda environment (e.g., named 'hw2')
conda create -n hw2 python=3.10

# 2) Activate the environment
conda activate hw2

# 3) Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 4) Install all other required packages
pip install jupyterlab notebook ipykernel transformers peft torchinfo tqdm scikit-learn matplotlib seaborn datasets
```

> If you plan to run notebooks interactively in VS Code Remote, ensure the **Jupyter** and **Python** extensions are installed on the remote, and the kernel points to the `hw2` environment.

---

## 2. How to Run

For both **Task1.ipynb** and **Task2.ipynb**, simply execute all cells in order from top to bottom.

The notebooks are designed to:
- Load and preprocess the datasets.
- Set up the models (Zero-Shot, Linear Probe, or LoRA).
- Run the training/evaluation loops.
- Automatically save the results as image files (e.g., training curves, prediction demonstrations) in the same directory.

---

## 3. How to Test Your Own Image

At the end of both **Task1.ipynb** and **Task2.ipynb**, there is a dedicated cell for testing your custom image (as required by the assignment).

**Steps:**
1. Upload your image file (e.g., `my_ntu_bird.jpg`) into the same folder as the `.ipynb` notebooks.
2. In the final code cell, find the following line:
   ```python
   YOUR_PHOTO_FILENAME = "my_ntu_bird.jpg"
   ```
3. Change the string to the exact filename of your uploaded image.
4. Run that cell. It will load your image, run the model(s), and generate a prediction result as a `.png` file in the same directory.

---

## Notes
- For reproducibility, consider fixing random seeds (e.g., `torch`, `numpy`, `random`) and documenting hyperparameters (batch size, learning rate, epochs, LoRA config).
- If `pandas.DataFrame.to_markdown()` raises a `tabulate` import error, either `pip install tabulate` or fallback to `df.to_string()` in the code.
