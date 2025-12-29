# Multimodal AI Final Project: PPT Image Detection & Manipulation

本專案是一個多模態 AI Pipeline，結合 **YOLOv8**（物件偵測）與 **SAM (Segment Anything Model)**（影像分割），目標是 **自動偵測並分割 PowerPoint 投影片中的插圖**，以支援後續的圖片編輯、去背與移動應用。

---

## 📋 目錄 (Table of Contents)

1. [專案概述 (Overview)](#專案概述-overview)
2. [環境安裝 (Installation)](#環境安裝-installation)
3. [資料準備 (Data Preparation)](#資料準備-data-preparation)
4. [模型訓練 (Training)](#模型訓練-training)
5. [訓練結果分析 (Evaluation)](#訓練結果分析-evaluation)
6. [Demo 與視覺化 (Visualization)](#demo-與視覺化-visualization)
7. [檔案結構 (File Structure)](#檔案結構-file-structure)

---

## 專案概述 (Overview)

本專案在 **RTX 4090 (24GB VRAM)** 環境下完成訓練與實驗，整體流程如下：

1. **Detection**
   使用 Fine-tuned **YOLOv8l** 偵測 PowerPoint 投影片中的插圖位置（Bounding Box）。

2. **Segmentation**
   將 YOLO 偵測到的 Bounding Box 作為 Prompt，輸入至 **SAM**，進行像素級影像分割與去背。

3. **Application**
   實現投影片插圖的自動擷取、移動與後續視覺處理應用。

---

## 環境安裝 (Installation)

本專案使用 **uv** 進行現代化 Python 套件與虛擬環境管理。

### 1️⃣ 初始化環境（Python 3.10）

```bash
uv init
uv python pin 3.10
```

### 2️⃣ 安裝核心依賴

```bash
# ultralytics: YOLOv8 框架
# segment-anything: Meta SAM 模型
# opencv-python: 影像處理
# matplotlib: 視覺化
uv add ultralytics opencv-python matplotlib segment-anything
```

### 3️⃣ （Optional）下載 SAM 權重

```bash
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## 資料準備 (Data Preparation)

原始資料為 **JSON 格式標註**，以 Pixel Coordinates 描述物件位置，並包含三種視圖：

* **Source**
* **Target**
* **Mask**

本專案使用自製腳本，將 JSON 標註轉換為 **YOLO Detection 格式**，並同時進行資料擴增，使資料量提升 **3 倍**。

### 3.1 執行資料轉換腳本

```bash
uv run convert_json_to_yolo_v3.py
```

此腳本會：

* 讀取 `annotations.json`
* 自動拆解 Source / Target / Mask
* 重新計算 Bounding Box
* 輸出 YOLO 訓練所需的 images / labels

---

### 3.2 建立 YOLO Dataset 設定檔

請確認資料已成功轉換至 `datasets/ppt_yolo_final/`，並設定 `ppt_final.yaml`：

```yaml
# ppt_final.yaml
path: /local/shankai/Multimodal/final/datasets/ppt_yolo_final  # 請確認為絕對路徑
train: train/images
val: val/images

nc: 1
names: ['slide_image']
```

---

## 模型訓練 (Training)

本專案針對 **RTX 4090** 進行訓練參數優化，以減少 CPU Data Loading 瓶頸並最大化 GPU 使用率。

### 執行訓練

```bash
uv run train.py
```

### 核心訓練參數（`train.py`）

* **Model**: `yolov8l.pt`
  使用 Large Model 以提升偵測準確率

* **Epochs**: `100`

* **Batch Size**: `32`
  針對 24GB VRAM 進行最佳化

* **Workers**: `16`
  加速 DataLoader

* **Cache**: `True`
  將影像快取至 RAM，大幅提升訓練速度

---

## 訓練結果分析 (Evaluation)

訓練完成後，結果會輸出至：

```text
runs/detect/ppt_finetune_v1/
```

（實際資料夾名稱依版本號而定）

### 關鍵評估指標（`results.png`）

* **Loss Curves**

  * `train/box_loss`
  * `train/cls_loss`
    應呈現穩定且平滑的下降趨勢

* **mAP Metrics**

  * **mAP50**：物件偵測準確率（理想值 > 0.95）
  * **mAP50-95**：高精度邊界準確率

### 分析結果

本模型在驗證集上達到：

* **mAP50 = 1.0**

顯示模型已能極為精準地學習 PPT 插圖的視覺特徵。

---

## Demo 與視覺化 (Visualization)

為期末報告與實驗展示，本專案提供多種視覺化腳本。

### 6.1 Ground Truth vs Prediction 對比

```bash
uv run visualize_comparison.py
```

輸出位置：

```text
runs/demo/comparison_results/
```

顏色說明：

* 🟩 **綠色框**：Ground Truth（人工標註）
* 🟥 **紅色框**：Model Prediction（模型預測）

判讀方式：

* 綠紅框重疊度（IoU）越高，代表模型越準確

---

### 6.2 SAM 分割 Pipeline Demo

```bash
uv run run_sam.py
```

此 Demo 展示完整流程：

1. YOLO 偵測 Bounding Box
2. SAM 生成精細 Mask
3. 去背並輸出可獨立使用的插圖素材

---

## 檔案結構 (File Structure)

```text
Multimodal_Final/
├── datasets/
│   ├── raw_images/             # 原始圖片 (Source / Target / Mask)
│   ├── annotations.json        # 原始 JSON 標註
│   └── ppt_yolo_final/         # [Generated] YOLO 格式訓練資料
├── runs/
│   ├── detect/                 # 訓練權重與 Log（best.pt 在此）
│   └── demo/                   # 視覺化輸出結果
├── convert_json_to_yolo_v3.py  # 資料處理與資料擴增腳本
├── train.py                    # YOLO 訓練腳本
├── visualize_comparison.py     # 預測結果對比腳本
├── run_sam.py                  # SAM 分割與去背 Pipeline
├── ppt_final.yaml              # YOLO Dataset 設定檔
└── README.md                   # 專案說明文件
```

---

## 📌 備註 (Notes)

* 請確認 `.gitignore` 已正確排除：

  * 訓練輸出 (`runs/`)
  * 權重檔 (`*.pt`, `*.pth`)
  * 私有資料集
* 若用於作品集或公開 Repo，請避免上傳原始標註資料與模型權重
