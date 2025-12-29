from ultralytics import YOLO
import cv2

# 1. 載入預訓練模型 (會自動下載 yolov8n.pt，這是最輕量版)
model = YOLO('yolov8n.pt')

# 2. 進行推論 (Inference)
# source 可以是圖片路徑
results = model.predict(source='test.jpeg', save=True, save_txt=True)

# 3. 顯示結果 (Optional)
# results 是一個 list，因為可能有多張圖
for result in results:
    boxes = result.boxes  # 抓出 bounding boxes
    print(f"偵測到 {len(boxes)} 個物件")
    
    # 這裡只是單純 print 出來看看格式
    for box in boxes:
        print(f"類別: {int(box.cls)}, 信心度: {float(box.conf):.2f}, 座標: {box.xyxy.tolist()}")

print("推論完成，結果已儲存在 runs/detect/predict 資料夾中")