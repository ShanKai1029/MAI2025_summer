import os
import json
import re
from tqdm import tqdm

import torch
from PIL import Image
import numpy as np

from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ===== 路徑設定 =====
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"

DATA_ROOT = "/local/shankai/Multimodal/hw3/data/vqa-rad"
TEST_JSON = os.path.join(DATA_ROOT, "vqa-rad-test.json")

# 你的 LoRA checkpoint，只要改成 v4 的實際路徑就好
LORA_PATH = "/local/shankai/Multimodal/hw3/checkpoints/llava-vqa-rad-lora/v4-20251124-034921/checkpoint-675"

MAX_NEW_TOKENS = 32


# ===== 一些工具函式（跟原本 eval.py 類似） =====
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def compute_bleu_scores(preds, gts):
    smoothie = SmoothingFunction().method4
    scores = []
    for p, g in zip(preds, gts):
        p_n = normalize_text(p)
        g_n = normalize_text(g)
        if not g_n.strip():
            continue
        score = sentence_bleu(
            [g_n.split()],
            p_n.split(),
            weights=(0.5, 0.5),
            smoothing_function=smoothie,
        )
        scores.append(score)
    if not scores:
        return 0.0, 0.0, 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def extract_qa(item):
    # 支援你現在 vqa-rad json 的格式（跟原本 eval.py 一樣）
    if "messages" in item:
        msgs = item["messages"]
        user = msgs[0]
        assistant = msgs[1]

        q_parts = []
        for c in user["content"]:
            if c["type"] == "text":
                q_parts.append(c["text"])
        q = " ".join(q_parts).strip()

        gt = assistant["content"][0]["text"].strip()
        return q, gt

    if "conversations" in item:
        convs = item["conversations"]
        human = convs[0]["value"]
        gt = convs[1]["value"].strip()
        if "<image>" in human:
            q = human.split("<image>")[-1].strip()
        else:
            q = human.strip()
        return q, gt

    raise ValueError("Unknown data format: no 'messages' or 'conversations'.")


# ===== 載入 base model + LoRA =====
def load_lora_model_and_processor():
    print("Loading base processor & model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    base_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Loading LoRA adapter from:", LORA_PATH)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    device = next(model.parameters()).device
    print("LoRA model loaded on", device)
    return model, processor, device


def main():
    # 讀 test json
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    model, processor, device = load_lora_model_and_processor()

    all_preds = []
    all_gts = []

    for item in tqdm(data, desc="Evaluating LoRA model on vqa-rad-test.json"):
        img_path = os.path.join(DATA_ROOT, item["image"])
        image = Image.open(img_path).convert("RGB")

        q, gt = extract_qa(item)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        pred = processor.decode(gen_ids, skip_special_tokens=True)

        all_preds.append(pred)
        all_gts.append(gt)

    mean_bleu, std_bleu, n = compute_bleu_scores(all_preds, all_gts)

    print("======== VQA-RAD test set BLEU (LoRA fine-tuned) ========")
    print(f"Checkpoint : {LORA_PATH}")
    print(f"Samples    : {n}")
    print(f"Mean BLEU-2: {mean_bleu:.4f}")
    print(f"Std  BLEU-2: {std_bleu:.4f}")


if __name__ == "__main__":
    main()
