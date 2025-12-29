# Multimodal VQA with LLaVA & Swift — README

## 1. Environment Setup

### 1.1 Create Conda Environment
```bash
conda create -n MAI_hw3 python=3.10
conda activate MAI_hw3
```

### 1.2 Install Requirements
```bash
pip install -r requirements.txt
```

### 1.3 Install Swift Framework
```bash
pip install "ms-swift[all]"
pip install msgspec
```
> *Note:* Some models require additional downloads through ModelScope during runtime.

---

## 2. Files Overview

### Training / Evaluation Scripts
| File | Description |
|------|-------------|
| `run_lora_vqarad.py` | Performs LoRA fine-tuning on VQA-RAD dataset using Swift SFT pipeline. |
| `eval_lora_vqarad.py` | Loads fine‑tuned LoRA adapter and evaluates with full test set. |
| `qualitative_vqarad_3samples.py` | Generates qualitative comparison (3 samples) between zero‑shot vs. fine‑tuned model. |

> **Important:** Paths inside scripts (`dataset`, `image root`, `checkpoint paths`) must be updated by the user based on their own folder structure.

---

## 3. Execution Order

### Step 1 — Download Dataset  
Place VQA‑RAD dataset in:
```
data/vqa-rad/
```
Expected structure:
```
vqa-rad-train.json
vqa-rad-test.json
images/
```

### Step 2 — Zero‑shot Evaluation (Optional)
```bash
python eval_zeroshot.py
```

### Step 3 — LoRA Fine‑tuning
```bash
python run_lora_vqarad.py
```
Outputs will be saved to:
```
checkpoints/llava-vqa-rad-lora/
```

### Step 4 — Quantitative Evaluation
```bash
python eval_lora_vqarad.py
```

### Step 5 — Qualitative Analysis (3 samples)
```bash
python qualitative_vqarad_3samples.py
```
Output images/tables will be saved under:
```
results/qualitative/
```

---

## 4. Notes

- Model checkpoint loading may be slow because LLaVA‑1.5‑7B downloads multiple shard files.
- VRAM usage can be adjusted using:
  - `batch_size`
  - `gradient_accumulation_steps`
  - `lora_rank`
- Ensure **ROOT_IMAGE_DIR** is correctly configured, or manually set in script.

---

## 5. Citation
If you use any models, cite:
- LLaVA  
- Swift (ModelScope)  
- VQA‑RAD dataset
