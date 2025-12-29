import cv2
import os
import glob
import torch
from ultralytics import YOLO
from pathlib import Path

# ================= 設定區 =================
# 1. 你的模型權重路徑
MODEL_PATH = "/local/shankai/Multimodal/final/runs/detect/ppt_finetune_v14/weights/best.pt"  
# (請確認 v13 是你最後訓練最好的那個版本資料夾)

# 2. 驗證集資料夾 (Val Images)
VAL_IMAGES_DIR = "/local/shankai/Multimodal/final/datasets/ppt_sorted/val/images"

# 3. 驗證集標註資料夾 (Val Labels)
VAL_LABELS_DIR = "/local/shankai/Multimodal/final/datasets/ppt_sorted/val/labels"

# 4. 輸出結果資料夾
OUTPUT_DIR = "/local/shankai/Multimodal/final/runs/demo/comparison_results"
# =========================================

def get_ground_truth_boxes(label_path, img_w, img_h):
    """讀取 .txt 標註檔並轉回像素座標"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        # YOLO 格式: class x_center y_center w h (Normalized)
        # 我們只需要座標，class 忽略
        x_c, y_c, w, h = map(float, parts[1:])
        
        # 轉回左上角 (x1, y1) 和右下角 (x2, y2) 像素座標
        x1 = int((x_c - w / 2) * img_w)
        y1 = int((y_c - h / 2) * img_h)
        x2 = int((x_c + w / 2) * img_w)
        y2 = int((y_c + h / 2) * img_h)
        boxes.append((x1, y1, x2, y2))
    return boxes

def main():
    # 1. 載入模型
    print(f"正在載入模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    
    # 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 取得所有驗證集圖片
    img_paths = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.*"))
    # 過濾非圖片檔
    img_paths = [f for f in img_paths if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"共找到 {len(img_paths)} 張驗證圖片，開始生成對比圖...")
    
    # 為了 Demo 方便，我們只隨機跑 20 張，或者你可以跑全部
    # img_paths = img_paths[:20] 

    for img_path in img_paths:
        filename = os.path.basename(img_path)
        stem = Path(img_path).stem
        
        # 讀取圖片
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        # --- A. 繪製 Ground Truth (標準答案) - 綠色 ---
        label_path = os.path.join(VAL_LABELS_DIR, stem + ".txt")
        gt_boxes = get_ground_truth_boxes(label_path, w, h)
        
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            # 畫綠框 (BGR: 0, 255, 0), 線寬 3
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, "Ground Truth", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- B. 繪製 Prediction (模型預測) - 紅色 ---
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        # 檢查有沒有預測出東西
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # 取得座標 (x1, y1, x2, y2)
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf)
                
                px1, py1, px2, py2 = coords
                # 畫紅框 (BGR: 0, 0, 255), 線寬 2
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
                
                label_text = f"Pred: {conf:.2f}"
                # 稍微錯開文字位置以免擋住 GT 文字
                cv2.putText(img, label_text, (px1, py2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # 如果沒預測到，印出提示
            cv2.putText(img, "No Detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- C. 存檔 ---
        save_path = os.path.join(OUTPUT_DIR, "compare_" + filename)
        cv2.imwrite(save_path, img)
        print(f"已儲存: {save_path}")

    print(f"\n全部完成！請去 {OUTPUT_DIR} 查看結果。")
    print("說明：\n[綠框] = 正確答案\n[紅框] = 模型預測")
    print("如果兩者重疊 -> 正確 (Good Case)")
    print("如果只有一個框或不重疊 -> 錯誤 (Bad Case)")

if __name__ == "__main__":
    main()