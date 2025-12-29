import os
import shutil
from pathlib import Path

# ================= 設定區 (請修改這裡) =================
# 1. 你的原始資料夾 (裡面混雜著圖片和txt，或者只有圖片)
SOURCE_DIR = "/local/shankai/Multimodal/final/datasets/ppt_sorted/train/images" 
# 注意：如果你之前的操作已經把圖片弄亂了，請指向現在圖片所在的資料夾

# 2. 如果你的 txt 檔在另一個資料夾，請填這裡；如果在同一個資料夾，填 None
SOURCE_LABEL_DIR = "/local/shankai/Multimodal/final/datasets/ppt_sorted/train/labels"

# 3. 目標輸出資料夾
OUTPUT_DIR = "/local/shankai/Multimodal/final/datasets/ppt_sorted"
# =======================================================

def main():
    # 建立目標資料夾結構
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    src_path = Path(SOURCE_DIR)
    lbl_path = Path(SOURCE_LABEL_DIR) if SOURCE_LABEL_DIR else src_path
    
    # 支援的圖片格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 計數器
    counts = {'train': 0, 'val': 0, 'skip': 0}
    
    print(f"正在掃描資料夾: {src_path} ...")
    
    files = list(src_path.iterdir())
    total_files = len(files)
    
    for i, file_path in enumerate(files):
        if file_path.suffix.lower() not in valid_exts:
            continue
            
        # 取得檔名 (不含副檔名)，例如 "image_005"
        stem = file_path.stem 
        
        # === 核心邏輯：依據結尾判斷 ===
        # 邏輯：看檔名最後一個字元，或是最後三碼
        # 根據你的描述：001~004 -> Train, 005 -> Val
        
        target_split = None
        
        # 這裡用 endswith 檢查字串結尾
        if stem.endswith('005'):
            target_split = 'val'
        elif stem.endswith(('001', '002', '003', '004')):
            target_split = 'train'
        else:
            # 如果檔名是 '006' 或其他，這裡會被跳過
            # 如果你想把所有剩下的都丟去 train，可以把下面的註解打開：
            # target_split = 'train' 
            pass

        if target_split:
            # 1. 複製圖片
            shutil.copy2(file_path, f"{OUTPUT_DIR}/{target_split}/images/{file_path.name}")
            
            # 2. 複製標註檔 (如果存在)
            txt_name = stem + ".txt"
            txt_src = lbl_path / txt_name
            
            if txt_src.exists():
                shutil.copy2(txt_src, f"{OUTPUT_DIR}/{target_split}/labels/{txt_name}")
            
            counts[target_split] += 1
        else:
            counts['skip'] += 1
            if counts['skip'] < 5: # 只印出前幾個被跳過的，避免洗版
                print(f"跳過檔案 (不符合規則): {file_path.name}")

        if i % 1000 == 0:
            print(f"進度: {i}/{total_files}...")

    print("\n" + "="*30)
    print("分類完成！")
    print(f"Train 圖片數: {counts['train']}")
    print(f"Val   圖片數: {counts['val']}")
    print(f"被跳過檔案數: {counts['skip']}")
    print(f"資料已儲存至: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    main()