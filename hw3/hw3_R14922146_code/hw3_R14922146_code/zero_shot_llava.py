import os
from typing import List, Dict

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 64

def load_model_and_processor():
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return model, processor, device

def ask_one(model, processor, device, image_path: str, question: str) -> str:
    image = Image.open(image_path).convert("RGB")

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
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(gen_ids, skip_special_tokens=True)
    return answer.strip()

def main():
    # TODO: 把這裡改成你自己的圖片路徑 & 問題
    samples: List[Dict] = [
        {
            "image": "zero_shot_images/01.jpeg",
            "questions": [
                "How many people in the image are pointing at the camera?",
                "What is the dominant color of the uniforms worn by the people?"
            ],
        },
        {
            "image": "zero_shot_images/02.jpeg",
            "questions": [
                "According to the poster, at which venue will the group perform?",
                "What represent the dates of the event listed on the poster?"
            ],
        },
        {
            "image": "zero_shot_images/03.jpeg",
            "questions": [
                "What is the promotional price value shown in large text for the ice cream?",
                "According to the top right section, on which days of the month is this promotion valid?"
            ],
        },
        {
            "image": "zero_shot_images/04.jpeg",
            "questions": [
                "Describe the weather conditions depicted in the background.",
                "What object is the character carrying on his back?"
            ],
        },
        {
            "image": "zero_shot_images/05.jpeg",
            "questions": [
                "Is the traffic in the image congested or flowing smoothly?",
                "What is the route number displayed on the LED sign of the bus in the middle?"
            ],
        },
    ]

    model, processor, device = load_model_and_processor()

    results = []
    for idx, s in enumerate(samples, start=1):
        img_path = os.path.abspath(s["image"])
        print(f"\n=== Image {idx}: {img_path} ===")
        for q_idx, q in enumerate(s["questions"], start=1):
            print(f"\nQ{idx}.{q_idx}: {q}")
            ans = ask_one(model, processor, device, img_path, q)
            print(f"A{idx}.{q_idx}: {ans}")
            results.append({
                "image": s["image"],
                "question": q,
                "answer": ans,
            })

    # 把結果存成 markdown，方便貼進報告
    out_path = "zero_shot_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| Image | Question | Answer |\n")
        f.write("|-------|----------|--------|\n")
        for r in results:
            # 把 | 換掉避免破壞表格
            q = r["question"].replace("|", "/")
            a = r["answer"].replace("|", "/")
            f.write(f"| {r['image']} | {q} | {a} |\n")

    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
