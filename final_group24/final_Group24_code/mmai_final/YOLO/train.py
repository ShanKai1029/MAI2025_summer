from ultralytics import YOLO

def main():
    # 1. 載入模型
    # 建議使用 yolov8m (Medium) 或 yolov8l (Large)
    # 因為你有 4090，跑大模型沒壓力，準度會更好
    model = YOLO('yolov8n.pt') 

    # 2. 訓練
    model.train(
        data='ppt_detect.yaml',
        epochs=25,             # 訓練 25 輪
        imgsz=640,              # 圖片大小
        batch=128,               # 4090 記憶體大，Batch 可以開大一點 (16, 32, 64 試試看)
        device=0,               # 指定使用 GPU 0 (你的 4090)
        name='ppt_finetune_v1', # 專案名稱
        workers=8               # 資料載入的執行緒
    )

if __name__ == '__main__':
    main()