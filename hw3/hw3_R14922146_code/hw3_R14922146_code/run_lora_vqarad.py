import os
import subprocess


# ====== 你可以調整的參數 ======
LORA_RANK = 8
CUDA_VISIBLE_DEVICES = "0"
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACC_STEPS = 4
MAX_LENGTH = 1024
MODEL_TYPE = "llava1_5_hf"
# ============================


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # VQA-RAD json 路徑
    dataset_dir = os.path.join(base_dir, "..", "data", "vqa-rad")
    train_json = os.path.join(dataset_dir, "vqa-rad-train.json")
    val_json = os.path.join(dataset_dir, "vqa-rad-test.json")

    if not os.path.exists(train_json):
        raise FileNotFoundError(f"找不到訓練集: {train_json}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"找不到驗證集: {val_json}")

    # ✅ 關鍵修正：ROOT_IMAGE_DIR 指到 vqa-rad 資料夾本身
    # 因為 json 裡的路徑是 "images/xxx.png"
    root_image_dir = dataset_dir
    if not os.path.isdir(root_image_dir):
        raise FileNotFoundError(
            f"找不到影像根目錄: {root_image_dir}\n"
            f"請確認 VQA-RAD 的資料夾位置是否正確。"
        )

    # checkpoint 輸出目錄
    output_dir = os.path.join(base_dir, "..", "checkpoints", "llava-vqa-rad-lora")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "swift", "sft",
        "--model", "llava-hf/llava-1.5-7b-hf",
        "--model_type", MODEL_TYPE,
        "--train_type", "lora",
        "--dataset", train_json,
        "--val_dataset", val_json,
        "--num_train_epochs", str(NUM_TRAIN_EPOCHS),
        "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),
        "--per_device_eval_batch_size", str(PER_DEVICE_BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRAD_ACC_STEPS),
        "--learning_rate", "1e-4",
        "--lora_rank", str(LORA_RANK),
        "--lora_alpha", "32",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--max_length", str(MAX_LENGTH),
        "--eval_steps", "100",
        "--save_steps", "200",
        "--save_total_limit", "2",
        "--logging_steps", "20",
        "--torch_dtype", "bfloat16",
        "--output_dir", output_dir,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    # ✅ 用環境變數告訴 swift：圖片根目錄在哪裡
    env["ROOT_IMAGE_DIR"] = root_image_dir

    print("即將執行指令：")
    print(" ".join(cmd))
    print(f"\n使用 GPU: {CUDA_VISIBLE_DEVICES}, LoRA rank = {LORA_RANK}")
    print(f"輸出會存到: {output_dir}")
    print(f"ROOT_IMAGE_DIR = {root_image_dir}\n")

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
