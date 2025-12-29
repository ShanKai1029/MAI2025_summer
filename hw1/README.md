### Installation

`pip install -r requirements.txt`

### Run

Before running `mae_pretrain.py`, for each different experiment, ensure that the relevant code sections are configured correctly. These sections are marked with a `# MODIFY` comment in the code.

* **1. Pre-training on CIFAR-10 (Baseline)**
    1.  `parser.add_argument`: Both `mask_strategy` and `decoder_depth` arguments should be commented out.
    2.  `run_name`: The dynamic `run_name` generation should be commented out.
    3.  `writer`: Use the line `writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))`.
    4.  `MAE_ViT`: Use the line `model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)`.
    5.  `CSV`: Use the line `csv_log_file = 'mae_pretrain_loss.csv'`.

* **2. Ablation Study: Decoder Depth**
    1.  `parser.add_argument`: The `decoder_depth` argument should be enabled/uncommented. It's best to comment out `mask_strategy`.
    2.  `run_name`: Use a line like `run_name = f"mae-pretrain_depth-{args.decoder_depth}"`.
    3.  `writer`: Use the line `writer = SummaryWriter(os.path.join('logs', 'cifar10', run_name))`.
    4.  `MAE_ViT`: Use the line `model = MAE_ViT(mask_ratio=args.mask_ratio, decoder_layer=args.decoder_depth).to(device)`.
    5.  `CSV`: Use a line like `csv_log_file = f'log_{run_name}.csv'`.

* **3. Ablation Study: Masking Strategy**
    1.  `parser.add_argument`: The `mask_strategy` argument should be enabled/uncommented. It's best to comment out `decoder_depth`.
    2.  `run_name`: Use a line like `run_name = f"mae-pretrain_mask-{args.mask_strategy}"`.
    3.  `writer`: Use the line `writer = SummaryWriter(os.path.join('logs', 'cifar10', run_name))`.
    4.  `MAE_ViT`: Use the line `model = MAE_ViT(mask_ratio=args.mask_ratio, mask_strategy=args.mask_strategy).to(device)`.
    5.  `CSV`: Use a line like `csv_log_file = f'log_{run_name}.csv'`.

### Visualization

* `visualize_predictions.py`: Generates the "Visualization of Classifier Predictions with MAE Pretraining".
* `visualize_reconstruction.py`: Generates "MAE reconstruction under different masking patterns".

_Note: You must edit these files to set the correct model path before running._

```bash
# Generate classifier prediction visualizations
python visualize_predictions.py

# Generate reconstruction visualizations
python visualize_reconstruction.py

```bash
# --- 1. Pre-training on CIFAR-10 (Baseline) ---

# Pretrain with MAE
python mae_pretrain.py

# Train classifier from scratch
python train_classifier.py --total_epoch 50 --output_model_path vit-t-classifier-from_scratch.pt

# Train classifier from pretrained model
python train_classifier.py --pretrained_model_path vit-t-mae.pt --output_model_path vit-t-classifier-from_pretrained.pt


# --- 2. Ablation Study: Decoder Depth ---

# Pretrain with MAE (where k can be 2, 4, or 8)
python mae_pretrain.py --decoder_depth k --model_path vit-mae-dk.pt

# Train classifier from pretrained model (where k can be 2, 4, or 8)
python train_classifier.py --pretrained_model_path vit-mae-dk.pt


# --- 3. Ablation Study: Masking Strategy ---

# Pretrain with MAE (where {choice} can be "random", "block", or "grid")
python mae_pretrain.py --mask_strategy {choice} --model_path vit-mae-mask-{choice}.pt

# Train classifier from pretrained model (where {choice} can be "random", "block", or "grid")
python train_classifier.py --pretrained_model_path vit-mae-mask-{choice}.pt
