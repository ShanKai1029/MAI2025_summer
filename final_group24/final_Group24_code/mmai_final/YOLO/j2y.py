import json
import os
import shutil
import cv2
from pathlib import Path

# ================= 設定區 (請根據你的實際路徑修改) =================
# 1. 你的 JSON 標註檔的路徑
JSON_FILE = "/local/shankai/Multimodal/final/train_meta.json" 
# (如果你的 json 檔名不一樣，請修改這裡)

# 2. 你的原始圖片所在的資料夾 (尚未分類的)
IMAGE_DIR = "/local/shankai/Multimodal/final/complete_slide/train"

# 3. 輸出結果的資料夾
OUTPUT_DIR = "/local/shankai/Multimodal/final/datasets/ppt_sorted"
# ===============================================================

def convert_bbox(bbox, img_w, img_h):
    """
    將 [xmin, ymin, xmax, ymax] 轉為 YOLO 格式 [x_center, y_center, w, h] (正規化 0~1)
    """
    x1, y1, x2, y2 = bbox
    
    # 計算寬高與中心
    box_w = x2 - x1
    box_h = y2 - y1
    x_c = (x1 + x2) / 2.0
    y_c = (y1 + y2) / 2.0
    
    # 正規化
    return (x_c / img_w, y_c / img_h, box_w / img_w, box_h / img_h)

def process_single_image(filename, bbox, split):
    """
    處理單張圖片：複製圖片 + 生成對應的 .txt 標註
    """
    src_path = Path(IMAGE_DIR) / filename
    
    if not src_path.exists():
        return False # 圖片不存在就跳過

    # 1. 讀取圖片大小 (為了正規化座標)
    # 我們只讀取 Header 資訊加速
    img = cv2.imread(str(src_path))
    if img is None: return False
    h, w, _ = img.shape
    
    # 2. 轉換座標
    yolo_bbox = convert_bbox(bbox, w, h)
    bbox_str = f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
    
    # 3. 複製圖片到目標資料夾
    dst_img_path = f"{OUTPUT_DIR}/{split}/images/{filename}"
    shutil.copy2(src_path, dst_img_path)
    
    # 4. 建立對應的 .txt 標註檔
    # 檔名要跟圖片一樣，只是副檔名改 .txt
    txt_filename = Path(filename).stem + ".txt"
    dst_txt_path = f"{OUTPUT_DIR}/{split}/labels/{txt_filename}"
    
    with open(dst_txt_path, 'w') as f:
        f.write(bbox_str + "\n")
        
    return True

def main():
    # 初始化資料夾結構
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    print(f"正在讀取 JSON: {JSON_FILE} ...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    print(f"找到 {len(samples)} 組資料，預計生成 {len(samples) * 3} 個標註檔案...")
    
    success_count = 0
    
    for item in samples:
        # 取得共用的 BBox
        bbox = item['bbox'] # [x1, y1, x2, y2]
        
        # 取得三個檔名
        files_to_process = [
            item.get('source'),
            item.get('target'),
            item.get('mask')
        ]
        
        # 判斷要分到 train 還是 val (看 source 的檔名結尾)
        # 邏輯：Source 檔名含 '005' 或 '_5' -> val，其餘 -> train
        # 這一組(source/target/mask)全部都去同一個地方，避免資料洩漏
        src_name = item.get('source', '')
        if '005' in src_name or '_5' in src_name:
            split = 'val'
        else:
            split = 'train'

        # 針對這三個檔案，分別執行處理
        for filename in files_to_process:
            if filename: # 確保檔名不是 None
                if process_single_image(filename, bbox, split):
                    success_count += 1

    print("="*40)
    print(f"處理完成！")
    print(f"總共生成: {success_count} 張訓練圖片與標註 (.txt)")
    print(f"輸出位置: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()