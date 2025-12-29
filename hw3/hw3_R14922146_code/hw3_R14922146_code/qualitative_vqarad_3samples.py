import os
import json
import argparse
from typing import Dict, List
import csv
import gc

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel

import matplotlib.pyplot as plt

MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 64


# ---------- helper: 問題 / 答案抽取 ----------

def extract_question(example: Dict) -> str:
    """
    從各種可能欄位裡抓出「問題文字」：
    - 常見欄位：question / query / prompt（不分大小寫）
    - key 名含 'ques' 或 'prompt'
    - swift 的 messages 結構
    """
    preferred_keys = ["question", "query", "prompt"]
    lower2orig = {k.lower(): k for k in example.keys()}

    # 1) 直接比常見 key
    for k in preferred_keys:
        if k in lower2orig:
            v = example[lower2orig[k]]
            return v if isinstance(v, str) else str(v)

    # 2) 模糊 key
    for k, v in example.items():
        if not isinstance(v, str):
            continue
        lk = k.lower()
        if "ques" in lk or "prompt" in lk:
            return v

    # 3) swift messages 格式
    if "messages" in example:
        msgs = example["messages"]
        if isinstance(msgs, list):
            for m in msgs:
                if m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, list):
                        texts = [
                            c.get("text", "")
                            for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        ]
                        if texts:
                            return texts[0]
                    elif isinstance(content, str):
                        return content
        # 如果真的沒抓到，就整個轉成字串
        return str(msgs)

    # 4) 從所有 string 欄位中找最像問句的
    candidate_strings = [v for v in example.values() if isinstance(v, str)]
    for s in candidate_strings:
        if "?" in s:
            return s
    if candidate_strings:
        return candidate_strings[0]

    print("extract_question() 無法在以下 keys 中找到問題欄位:", list(example.keys()))
    raise KeyError("找不到 question / query 欄位，請檢查 json。")


def extract_answer(example: Dict) -> str:
    """
    嘗試抓出 ground truth answer：
    - 常見欄位：answer / response / label / gt_answer（不分大小寫）
    - key 名含 'ans' 或 'label'
    - swift messages 裡最後一個 assistant
    """
    lower2orig = {k.lower(): k for k in example.keys()}

    # 1) 常見 key
    preferred_keys = ["answer", "response", "label", "gt_answer"]
    for k in preferred_keys:
        if k in lower2orig:
            v = example[lower2orig[k]]
            return v if isinstance(v, str) else str(v)

    # 2) 模糊 key
    for k, v in example.items():
        if not isinstance(v, str):
            continue
        lk = k.lower()
        if "ans" in lk or "label" in lk:
            return v

    # 3) swift messages 格式
    if "messages" in example:
        msgs = example["messages"]
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    content = m.get("content")
                    if isinstance(content, list):
                        texts = [
                            c.get("text", "")
                            for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        ]
                        if texts:
                            return texts[0]
                    elif isinstance(content, str):
                        return content
        return str(msgs)

    # 找不到就 N/A
    return "N/A"


def get_image_path(example: Dict, image_root: str) -> str:
    # 你的 vqa-rad json 應該有欄位 "image": "images/train_00001.png"
    if "image" in example:
        rel = example["image"]
    elif "image_path" in example:
        rel = example["image_path"]
    else:
        raise KeyError("無法在 example 中找到 image / image_path 欄位")

    return os.path.join(image_root, rel)


# ---------- 推論 ----------

def generate_answer(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    image: Image.Image,
    question: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
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
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    pred = processor.decode(gen_ids, skip_special_tokens=True)
    return pred.strip()


# ---------- 輸出：CSV / Markdown / PNG 表格 ----------

def save_csv(results: List[Dict], path: str):
    fieldnames = ["idx", "image", "question", "gt", "base_pred", "lora_pred"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "idx": item["idx"],
                    "image": os.path.basename(item["image_path"]),
                    "question": item["question"].replace("\n", " "),
                    "gt": item["gt"].replace("\n", " "),
                    "base_pred": item["base_pred"].replace("\n", " "),
                    "lora_pred": item["lora_pred"].replace("\n", " "),
                }
            )


def save_markdown(results: List[Dict], path: str):
    lines = []
    lines.append("| Idx | Image | Question | Ground Truth | w/o FT (Base) | w/ FT (LoRA) |")
    lines.append("|-----|-------|----------|--------------|---------------|--------------|")
    for item in results:
        idx = item["idx"]
        img_name = os.path.basename(item["image_path"])
        q = item["question"].replace("\n", " ")
        gt = item["gt"].replace("\n", " ")
        base_pred = item["base_pred"].replace("\n", " ")
        lora_pred = item["lora_pred"].replace("\n", " ")

        def short(s, L=60):
            return (s[: L - 3] + "...") if len(s) > L else s

        line = f"| {idx} | {img_name} | {short(q)} | {short(gt)} | {short(base_pred)} | {short(lora_pred)} |"
        lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_png_table(results: List[Dict], path: str):
    headers = ["Idx", "Image", "Question", "GT", "Base", "LoRA"]
    table_data = [headers]

    def short(s, L=45):
        s = s.replace("\n", " ")
        return (s[: L - 3] + "...") if len(s) > L else s

    for item in results:
        row = [
            str(item["idx"]),
            os.path.basename(item["image_path"]),
            short(item["question"]),
            short(item["gt"]),
            short(item["base_pred"]),
            short(item["lora_pred"]),
        ]
        table_data.append(row)

    n_rows = len(table_data)
    n_cols = len(headers)

    fig, ax = plt.subplots(figsize=(3 * n_cols, 0.6 * n_rows + 1))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- 新增：存圖片 & 拼成一張 grid 圖 ----------

def save_selected_images(results: List[Dict], img_dir: str):
    os.makedirs(img_dir, exist_ok=True)
    for i, item in enumerate(results):
        img = Image.open(item["image_path"]).convert("RGB")
        base_name = os.path.basename(item["image_path"])
        out_name = f"sample_{i}_{item['idx']}_{base_name}"
        out_path = os.path.join(img_dir, out_name)
        img.save(out_path)
        item["saved_image_path"] = out_path  # 之後畫 grid 用


def save_image_grid(results: List[Dict], grid_path: str):
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, results):
        img_path = item.get("saved_image_path", item["image_path"])
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"idx={item['idx']}", fontsize=10)

    plt.tight_layout()
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/local/shankai/Multimodal/hw3/data/vqa-rad/vqa-rad-test.json",
        help="VQA-RAD 的 test JSON 路徑",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/local/shankai/Multimodal/hw3/data/vqa-rad",
        help="圖片根目錄（json 裡的 image 欄位會接在這後面）",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="/local/shankai/Multimodal/hw3/checkpoints/llava-vqa-rad-lora/v4-20251124-034921/checkpoint-675",
        help="LoRA checkpoint 資料夾",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[10, 123, 300],
        help="要抽的 sample index，例如: --indices 10 123 300",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="vqarad_qualitative_3samples.csv",
        help="輸出 CSV 檔名",
    )
    parser.add_argument(
        "--md_path",
        type=str,
        default="vqarad_qualitative_3samples.md",
        help="輸出 Markdown 檔名",
    )
    parser.add_argument(
        "--png_path",
        type=str,
        default="vqarad_qualitative_3samples.png",
        help="輸出 PNG 表格檔名",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="vqarad_qualitative_imgs",
        help="把抽出的圖片另存到這個資料夾",
    )
    parser.add_argument(
        "--grid_path",
        type=str,
        default="vqarad_qualitative_3samples_grid.png",
        help="三張圖拼在一起的 grid 圖片檔名",
    )
    args = parser.parse_args()

    # 讀 dataset
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 處理 indices，避免超出範圍
    indices: List[int] = []
    for i in args.indices:
        if 0 <= i < len(data):
            indices.append(i)
        else:
            print(f"[WARN] index {i} 超出資料長度 {len(data)}，略過。")
    if not indices:
        raise ValueError("沒有合法的 indices，可以重新指定 --indices")

    # 1. 載 base model
    print("=== Loading base Llava model ===")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model_base = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model_base.eval()
    device = next(model_base.parameters()).device
    print("Base model device:", device)

    results = []

    # 先用 base model 跑
    for idx in indices:
        ex = data[idx]
        img_path = get_image_path(ex, args.image_root)
        image = Image.open(img_path).convert("RGB")
        q = extract_question(ex)
        gt = extract_answer(ex)

        base_pred = generate_answer(
            model_base,
            processor,
            device,
            image,
            q,
        )

        results.append(
            {
                "idx": idx,
                "image_path": img_path,
                "question": q,
                "gt": gt,
                "base_pred": base_pred,
                "lora_pred": None,
            }
        )

    # 釋放 base model
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. 載 LoRA model
    print("=== Loading LoRA fine-tuned model ===")
    model_for_lora = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model_lora = PeftModel.from_pretrained(model_for_lora, args.lora_dir)
    model_lora.eval()
    device_lora = next(model_lora.parameters()).device
    print("LoRA model device:", device_lora)

    # 用 LoRA model 跑同樣 samples
    for item in results:
        image = Image.open(item["image_path"]).convert("RGB")
        q = item["question"]

        lora_pred = generate_answer(
            model_lora,
            processor,
            device_lora,
            image,
            q,
        )
        item["lora_pred"] = lora_pred

    # 3. 存成 CSV / Markdown / PNG 表格
    save_csv(results, args.csv_path)
    save_markdown(results, args.md_path)
    save_png_table(results, args.png_path)

    # 4. 另外把實際圖片也存出來 + 拼一張 grid
    save_selected_images(results, args.img_dir)
    save_image_grid(results, args.grid_path)

    print(f"\n✅ 已輸出：")
    print(f"  - CSV：{args.csv_path}")
    print(f"  - Markdown：{args.md_path}")
    print(f"  - 表格圖片：{args.png_path}")
    print(f"  - 個別圖片資料夾：{args.img_dir}/")
    print(f"  - 三張圖拼在一起的 grid：{args.grid_path}")


if __name__ == "__main__":
    main()
