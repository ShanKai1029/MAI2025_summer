from ultralytics import YOLO
import cv2
import os

def main():
    # ---------------------------------------------------------
    # 1. 設定路徑
    # ---------------------------------------------------------
    # 你的模型路徑 (請確認路徑正確)
    model_path = 'runs/detect/single_test3/weights/best.pt'
    
    # 你想測試的圖片 (我們先用剛剛那張 ppt_001.png 來驗證它有沒有學會)
    image_path = 'datasets/dataset_single/val/images/ppt_001.jpeg'

    # ---------------------------------------------------------
    # 2. 載入訓練好的模型
    # ---------------------------------------------------------
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型檔案 -> {model_path}")
        return

    print(f"正在載入模型: {model_path} ...")
    model = YOLO(model_path)

    # ---------------------------------------------------------
    # 3. 進行預測 (Inference)
    # ---------------------------------------------------------
    print(f"正在偵測圖片: {image_path} ...")
    # conf=0.25 代表信心度大於 25% 才算找到
    results = model.predict(source=image_path, save=True, conf=0.25)

    # ---------------------------------------------------------
    # 4. 取得座標並裁切圖片 (為了你的下一階段任務)
    # ---------------------------------------------------------
    # 讀取原圖
    img = cv2.imread(image_path)
    
    # 取得偵測結果
    result = results[0]
    
    # 如果有偵測到物件
    if len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            # 取得座標 (x1, y1, x2, y2)
            # xyxy[0] 是一個 tensor，轉成 cpu numpy array
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            
            print(f"找到目標 #{i+1}，座標: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # 裁切圖片 (注意 OpenCV 是 [y:y+h, x:x+w])
            cropped_img = img[y1:y2, x1:x2]
            
            # 存檔裁切後的圖片
            crop_filename = f"crop_result_{i}.jpg"
            cv2.imwrite(crop_filename, cropped_img)
            print(f"已將裁切圖片存為: {crop_filename}")
            
    else:
        print("嗚嗚，什麼都沒偵測到 (可能是訓練資料太少或信心度設太高)")

if __name__ == '__main__':
    main()