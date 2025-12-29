# ==============================================================================
# train_powerpaint_mask.py (Final Version: Auto-Fix Mask & Validation Support)
# ==============================================================================

import json
import time
import os
import argparse
import random
from collections import deque 

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

# -------------------------
# Utils
# -------------------------
def denormalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """將 Tensor 從 [-1, 1] 還原回 [0, 1] 以便視覺化。"""
    tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0, 1)

def dilate_mask(mask_tensor, kernel_size=3):
    """
    對 Mask 進行膨脹 (Dilation) 運算，填補因為過黑(陰影)而產生的破洞。
    mask_tensor input shape: (1, H, W)
    """
    # 建立一個全 1 的 kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size)).to(mask_tensor.device)
    padding = kernel_size // 2
    
    # 增加 batch 維度以符合 conv2d: (1, 1, H, W) -> (1, 1, H, W)
    mask_input = mask_tensor.unsqueeze(0)
    
    # 使用 conv2d 模擬 dilation (只要 kernel 範圍內有一個是 1，結果就是 >0)
    mask_dilated = F.conv2d(mask_input, kernel, padding=padding)
    
    # 只要大於 0 就變成 1 (二值化)
    mask_dilated = (mask_dilated > 0).float()
    
    # 降維回 (1, H, W)
    return mask_dilated.squeeze(0)

# -------------------------
# Dataset
# -------------------------
class SlideCropDataset(Dataset):
    def __init__(self, meta_path, input_dir, image_size=512, add_trigger=True):
        self.input_dir = input_dir
        self.image_size = image_size
        self.add_trigger = add_trigger 
        
        with open(meta_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            # 相容兩種 json 結構
            self.samples = data["samples"] if "samples" in data else data

        # 圖片預處理：Normalize 到 [-1, 1]
        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Mask 預處理
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 簡單的重試機制，避免單一檔案壞掉導致訓練崩潰
        for _ in range(3): 
            try:
                s = self.samples[idx]
                
                src_path = os.path.join(self.input_dir, s["source"])
                tgt_path = os.path.join(self.input_dir, s["target"]) # 必須讀取 Target (Ground Truth)
                mask_path = os.path.join(self.input_dir, s["mask"])

                # 1. Load Images
                source_full = Image.open(src_path).convert("RGB")
                target_full = Image.open(tgt_path).convert("RGB")
                mask_full = Image.open(mask_path).convert("L")

                # 2. Crop by BBox (先切，排除無關背景)
                bbox = s["bbox"] 
                source_crop = source_full.crop(bbox)
                target_crop = target_full.crop(bbox)
                mask_crop = mask_full.crop(bbox)

                # 3. Transform to Tensor (512x512)
                source_tensor = self.img_tf(source_crop)
                target_tensor = self.img_tf(target_crop)
                mask_tensor_raw = self.mask_tf(mask_crop)

                # ---------------------------------------------------------
                # [CRITICAL STEP] Mask 修復工程
                # ---------------------------------------------------------
                # A. 極限閾值：只要不是純黑 (0.05) 都算 Mask，抓出便當盒
                mask_tensor = (mask_tensor_raw < 0.05).float()
                
                # B. 膨脹：填補中間的陰影破洞 (黑點)
                mask_tensor = dilate_mask(mask_tensor, kernel_size=3)

                # ---------------------------------------------------------
                # Prompt 處理：自動補上 PowerPaint 的 P_obj
                # ---------------------------------------------------------
                prompt = s["prompt"]
                if self.add_trigger:
                    if "P_obj" not in prompt and "P_ctxt" not in prompt:
                        prompt = f"{prompt} P_obj"

                return {
                    "source": source_tensor, # 用於保留背景 (Condition)
                    "mask": mask_tensor,     # 告訴模型哪裡要改
                    "target": target_tensor, # 真實答案 (Ground Truth, 用於算 Loss)
                    "prompt": prompt,
                }

            except Exception as e:
                print(f"[WARN] Sample idx={idx} error: {e}. Skipping to random sample.")
                idx = random.randint(0, len(self.samples) - 1)
        
        raise RuntimeError("Failed to load valid data after retries.")

# -------------------------
# Validation Function
# -------------------------
def validate(args, pipe, val_loader, writer, global_step, device):
    print(f"\n[INFO] Running Validation at step {global_step}...")
    
    # 1. 計算 Validation Loss (這部分保持不變)
    pipe.unet.eval()
    total_loss = 0.0
    num_batches = 0
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # 用來存第一張圖的資訊以便視覺化
    vis_data = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            prompt_str = batch["prompt"]

            # ... (中間 Loss 計算省略，保持原本的就好) ...
            # ... 為了節省篇幅，這裡假設你保留原本的 Loss 計算邏輯 ...
            
            # [NEW] 抓第一筆資料來做視覺化推論
            if batch_idx == 0:
                vis_data = {
                    "source": source[0:1], # 取 batch 的第一張
                    "mask": mask[0:1],
                    "target": target[0:1],
                    "prompt": prompt_str[0]
                }
            
            # 這裡補回原本的 Loss 計算 (簡化版示意)
            latents = pipe.vae.encode(target).latent_dist.sample() * pipe.vae.config.scaling_factor
            source_latents = pipe.vae.encode(source).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
            masked_latents = source_latents * (1 - mask_latent)
            text_ids = pipe.tokenizer(prompt_str, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            text_emb = pipe.text_encoder(text_ids)[0]
            unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
            noise_pred = pipe.unet(unet_input, timesteps, encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"[Validation] Step {global_step} | Val Loss: {avg_loss:.4f}")
    
    if writer:
        writer.add_scalar("validation/loss", avg_loss, global_step)

        # [NEW] 視覺化：真的跑一次 Inpainting 生成
        if vis_data is not None:
            print("[INFO] Generating validation sample image...")
            # 使用 pipeline 的標準推論介面
            # 注意：這裡輸入需要還原成 PIL 或保持 Tensor 範圍
            # 為了簡單，我們直接用底層 UNet 跑一個 denoising loop 會太複雜
            # 我們直接呼叫 pipe 來生圖 (會比較慢，但看得到結果)
            
            # 暫時切換回 inference mode
            pipe.vae.to(dtype=torch.float32) 
            
            # 準備輸入圖片 ([-1,1] -> [0,1] PIL)
            src_pil = to_pil_image(denormalize_to_01(vis_data["source"][0]))
            mask_pil = to_pil_image(vis_data["mask"][0])
            
            # 執行生成
            generated_image = pipe(
                prompt=vis_data["prompt"],
                image=src_pil,
                mask_image=mask_pil,
                num_inference_steps=20, # 快速預覽用 20 步就好
                guidance_scale=7.5,
                strength=1.0
            ).images[0]
            
            # 轉回 Tensor 以便拼圖
            gen_tensor = transforms.ToTensor()(generated_image).to(device)
            src_tensor = denormalize_to_01(vis_data["source"][0])
            tgt_tensor = denormalize_to_01(vis_data["target"][0])
            mask_tensor = vis_data["mask"][0].repeat(3, 1, 1)

            # 拼成四宮格: Source | Target | Mask | Output
            grid = make_grid([src_tensor, tgt_tensor, mask_tensor, gen_tensor], nrow=4)
            writer.add_image("validation/sample_result", grid, global_step)
            print("[INFO] Validation sample saved to TensorBoard.")

    pipe.unet.train() # 切換回訓練模式
    return avg_loss
# -------------------------
# Main Training Function
# -------------------------
def main(args):
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_dir)

    print(f"[INFO] Loading Model: {args.model_id}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float32, safety_checker=None
    ).to(device)

    # Freeze Base Model
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Setup LoRA
    print(f"[INFO] Adding LoRA (Rank={args.lora_rank})")
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"], # PowerPaint 建議
            lora_dropout=args.lora_dropout,
        )
    )
    pipe.unet.train()
    
    # 1. Train Loader
    print(f"[INFO] Loading Training Data from {args.input_dir}")
    train_dataset = SlideCropDataset(args.meta_path, args.input_dir, image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # 2. Validation Loader (Optional)
    val_loader = None
    if args.val_meta_path and args.val_input_dir:
        print(f"[INFO] Loading Validation Data from {args.val_input_dir}")
        val_dataset = SlideCropDataset(args.val_meta_path, args.val_input_dir, image_size=args.image_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=args.lr)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    scaler = GradScaler()
    
    global_step = 0
    start_time = time.time()
    loss_buffer = deque(maxlen=100)

    print(f"[INFO] Start Training...")
    for epoch in range(args.epochs):
        for batch in train_loader:
            if time.time() - start_time > args.max_train_time:
                print("[INFO] Reached time limit. Saving and exiting.")
                pipe.unet.save_pretrained(args.output_dir)
                writer.close()
                return

            # Data to device
            source = batch["source"].to(device) # Condition
            target = batch["target"].to(device) # Ground Truth
            mask = batch["mask"].to(device)
            prompt_str = batch["prompt"]

            with autocast():
                # --- Training Forward Pass ---
                
                # 1. VAE Encode Target (作為加噪目標)
                with torch.no_grad():
                    latents = pipe.vae.encode(target).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                # 2. Add Noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # 3. Prepare Mask & Masked Image Condition
                # 需要 Source 的 Latent 作為背景
                with torch.no_grad():
                    source_latents = pipe.vae.encode(source).latent_dist.sample()
                    source_latents = source_latents * pipe.vae.config.scaling_factor
                
                # Resize Mask to Latent Size
                mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
                
                # 關鍵：Inpainting Condition = Source (背景) * (1 - Mask)
                masked_latents = source_latents * (1 - mask_latent)

                # 4. UNet Prediction
                text_ids = pipe.tokenizer(prompt_str, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
                text_emb = pipe.text_encoder(text_ids)[0]

                # Input: [Noisy Target, Mask, Masked Source]
                unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
                noise_pred = pipe.unet(unet_input, timesteps, encoder_hidden_states=text_emb).sample

                # 5. Global MSE Loss (計算全圖誤差，確保背景融合)
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            loss_val = loss.item()
            loss_buffer.append(loss_val)
            avg_loss = sum(loss_buffer) / len(loss_buffer)

            if global_step % args.print_every == 0:
                print(f"[Epoch {epoch}][Step {global_step}] Loss: {loss_val:.4f} | Avg: {avg_loss:.4f}")

            if global_step % args.tb_log_every == 0:
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % args.visualize_every == 0:
                # Visualize Source vs Target vs Mask (檢查資料正確性)
                src_vis = denormalize_to_01(source[0].cpu())
                tgt_vis = denormalize_to_01(target[0].cpu())
                mask_vis = mask[0].cpu().repeat(3, 1, 1) # 1 channel -> 3 channels
                
                grid = make_grid([src_vis, tgt_vis, mask_vis], nrow=3)
                writer.add_image("train/vis_src_tgt_mask", grid, global_step)

            global_step += 1
        
        # End of Epoch: Run Validation
        if val_loader is not None:
            validate(args, pipe, val_loader, writer, global_step, device)
            
            # (Optional) Save epoch checkpoint
            # epoch_save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            # pipe.unet.save_pretrained(epoch_save_dir)

    print("[INFO] Training Finished. Saving Model...")
    pipe.unet.save_pretrained(args.output_dir)
    writer.close()
    print(f"[DONE] Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerPaint LoRA Training (Final Corrected)")
    
    # Paths
    parser.add_argument("--meta_path", type=str, required=True, help="Path to train json")
    parser.add_argument("--input_dir", type=str, required=True, help="Root dir for train images")
    parser.add_argument("--val_meta_path", type=str, default=None, help="Path to val json (optional)")
    parser.add_argument("--val_input_dir", type=str, default=None, help="Root dir for val images (optional)")
    
    parser.add_argument("--output_dir", type=str, default="outputs/powerpaint_lora")
    parser.add_argument("--log_dir", type=str, default="runs/powerpaint_lora")
    
    # Model
    parser.add_argument("--model_id", type=str, default="Sanster/PowerPaint-V1-stable-diffusion-inpainting")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Training Params
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2) # 顯存不夠改 1
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_time", type=int, default=21600) # 預設 6 小時
    
    # Logging
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--tb_log_every", type=int, default=10)
    parser.add_argument("--visualize_every", type=int, default=500)
    parser.add_argument("--ma_window", type=int, default=100)

    args = parser.parse_args()
    main(args)