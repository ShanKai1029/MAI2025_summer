from datasets import load_dataset
import json
import os

# 下載 HF 上的 VQA-RAD
ds = load_dataset("flaviagiammarino/vqa-rad")

root = "/local/shankai/Multimodal/hw3/data/vqa-rad"
img_dir = os.path.join(root, "images")
os.makedirs(root, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

def convert_split(split):
    out = []
    for i, ex in enumerate(ds[split]):
        img = ex["image"]               # HF 幫你變成 PIL Image 物件
        img_name = f"{split}_{i:05d}.png"
        img_path = os.path.join(img_dir, img_name)
        img.save(img_path)

        q = ex["question"]
        a = ex["answer"]

        sample = {
            "image": os.path.join("images", img_name),
            "conversations": [
                {"from": "human", "value": "<image>\n" + q},
                {"from": "gpt",   "value": a}
            ]
        }
        out.append(sample)
    return out

train_data = convert_split("train")
test_data  = convert_split("test")

with open(os.path.join(root, "vqa-rad-train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(root, "vqa-rad-test.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("Done! Saved to", root)
