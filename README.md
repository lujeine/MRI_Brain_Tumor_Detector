# Brain Tumor MRI Detector

A small PyTorch CNN that classifies brain MRI slices as **tumor** or **healthy**.

Dataset: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) (not included; download it and unzip so that `brain_tumor_dataset/yes` and `brain_tumor_dataset/no` are in the repo root).

## Run

```bash
pip install -r requirements.txt
jupyter notebook TumorDetector.ipynb
```

## What the notebook does

1. Loads images, skips unreadable files, removes exact duplicates.
2. Stratified 70/15/15 train/val/test split (normalisation stats from train only).
3. Trains with augmentation, dropout, weight decay and early stopping on validation loss.
4. Evaluates once on the held-out test set (accuracy with 95% CI, precision, recall, F1, ROC AUC, confusion matrix).
5. Overfitting checks: train-vs-test gap, stratified 5-fold CV, shuffled-label test.
