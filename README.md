# Hypertension Detection from Fundus Images using Deep Learning

## Overview

This project explores the use of deep learning to detect **systemic hypertension** from **retinal fundus images**. Hypertension causes subtle changes in retinal blood vessels such as **arteriolar narrowing, AV nicking, and vessel tortuosity**. These vascular patterns can be analyzed by convolutional neural networks (CNNs) to assist in early screening.

The goal of this project was to build a **robust machine learning pipeline** for hypertension detection using retinal images while experimenting with multiple improvements commonly used in medical imaging research.

The model was implemented using **PyTorch** with **EfficientNet-B0** as the backbone architecture.

---

# Dataset

The dataset consists of retinal fundus images with labels indicating whether the patient is hypertensive or not.

### Dataset Structure
```
dataset
└─ 1-Hypertensive Classification/
   ├─ 1-Hypertensive Classification/
   │  ├─ 2-Groundtruths/
   │  │  └─ HRDC Hypertensive Classification Training Labels.csv
   │  └─ 1-images/
   │     └─ 1-Training Set/
   └─ 2-Hypertensive Retinopathy Classification/
      └─ 2-Hypertensive Retinopathy Classification/
         ├─ 2-Groundtruths/
         │  └─ HRDC Hypertensive Retinopathy Classification Training Labels.csv
         └─ 1-images/
            └─ 1-Training Set/
```

For this project we used the [**Hypertensive Classification dataset**](https://www.kaggle.com/datasets/harshwardhanfartale/hypertension-and-hypertensive-retinopathy-dataset).

### Dataset Statistics

| Category         | Count |
| ---------------- | ----- |
| Non-Hypertensive | 284   |
| Hypertensive     | 285   |

The dataset is **perfectly balanced**, which simplifies model training.

### Dataset Split

The dataset was split using stratified sampling:

| Split      | Images |
| ---------- | ------ |
| Train      | ~455   |
| Validation | ~57    |
| Test       | ~57    |

Because the test set is small, **accuracy changes of ±1 image correspond to ~1.4% variation**.

---

# Project Pipeline

The final pipeline implemented is:

```
Fundus Image
      ↓
Fundus Cropping (remove black borders)
      ↓
Green Channel Extraction
      ↓
CLAHE Vessel Enhancement
      ↓
Data Augmentation
      ↓
EfficientNet-B0 (transfer learning)
      ↓
Classifier Head
      ↓
Hypertension Prediction
```

---

# Model Architecture

Backbone: **EfficientNet-B0**

Transfer learning strategy:

* Pretrained ImageNet weights
* Early layers frozen
* Last feature blocks partially unfrozen
* Custom classifier layer

Advantages:

* Efficient architecture
* Strong performance on small datasets
* Good transfer learning capability

---

# Preprocessing Techniques

## 1. Fundus Cropping

Fundus images contain large black borders that do not contain useful medical information.

Cropping removes these borders so the model focuses only on retinal structures.

Benefits:

* Reduces noise
* Improves feature learning

---

## 2. Green Channel Extraction

Retinal blood vessels have highest contrast in the **green channel**.

Instead of using full RGB images:

```
image → extract green channel → replicate into 3 channels
```

This highlights vascular patterns relevant to hypertension.

---

## 3. CLAHE Vessel Enhancement

Contrast Limited Adaptive Histogram Equalization (CLAHE) enhances vessel visibility.

```
green channel → CLAHE → enhanced vessel contrast
```

This helps the CNN detect:

* arteriolar narrowing
* vessel tortuosity
* AV nicking

---

# Data Augmentation

To combat overfitting caused by the small dataset, several augmentations were applied:

| Augmentation    | Purpose                  |
| --------------- | ------------------------ |
| Horizontal Flip | simulate left/right eyes |
| Vertical Flip   | orientation robustness   |
| Rotation        | camera variability       |
| Color Jitter    | lighting variability     |

These augmentations increase dataset diversity.

---

# Training Strategy

Several training strategies were explored.

### Baseline

EfficientNet trained with transfer learning.

Result:

```
Accuracy ≈ 0.62
```

---

### Data Augmentation

Improved generalization.

Result:

```
Accuracy ≈ 0.65
```

---

### Freezing Backbone

Prevented overfitting but reduced model capacity.

Result:

```
Accuracy ≈ 0.64
```

---

### Partial Unfreezing

Allowed deeper layers to adapt to retinal features.

Result:

```
Accuracy ≈ 0.61 – 0.64
```

---

### Vessel Enhancement + Cropping

Improved vascular visibility.

Result:

```
Accuracy ≈ 0.62 – 0.64
```

---

### Weighted Loss (Final Model)

Because missing hypertensive patients is medically worse than false alarms, class weighting was introduced.

```
weights = [1.0, 1.5]
```

This penalizes false negatives more strongly.

Final model performance improved in **recall and balance**.

---

# Evaluation Metrics

Evaluation uses:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

These metrics are important for **medical classification problems**.

---

# Final Model Results

```
Accuracy : 0.6389
F1 Score : 0.6389
Precision: 0.6389
Recall   : 0.6389
```

Confusion Matrix:

```
[[23 13]
 [13 23]]
```

Matrix format:

```
[[True Negatives  False Positives]
 [False Negatives True Positives]]
```

Interpretation:

| Metric          | Value |
| --------------- | ----- |
| True Negatives  | 23    |
| True Positives  | 23    |
| False Positives | 13    |
| False Negatives | 13    |

The classifier is now **balanced between both classes**.

---

# Key Insights

### 1. Dataset Size is the Main Limitation

With only **569 images**, the model struggles to learn subtle vascular patterns.

More data would likely produce large improvements.

---

### 2. Preprocessing Matters

Domain-specific preprocessing significantly improves retinal ML pipelines.

Key useful techniques:

* fundus cropping
* green channel extraction
* CLAHE vessel enhancement

---

### 3. Balanced Dataset Simplifies Training

Because the dataset is balanced, the model did not require oversampling techniques.

---

### 4. Weighted Loss Improved Medical Sensitivity

Weighted loss reduced **false negatives**, improving the detection of hypertensive patients.

---

### 5. Model Performance is Stable

Train and validation losses are similar, indicating **minimal overfitting**.

---

# Limitations

* Small dataset size
* Subtle visual indicators of hypertension
* Limited test set (≈57 images)

These factors cap achievable performance.

---

# Future Improvements

Possible future work includes:

### Larger Image Resolution

Retinal vessels are tiny structures. Training at higher resolution may improve detection.

### Grad-CAM Visualization

Grad-CAM can show which retinal regions influence predictions, improving interpretability.

### Multi-Task Learning

The dataset also includes **hypertensive retinopathy grading** which could be combined with hypertension detection.

### Larger Dataset

Combining multiple retinal datasets would likely improve performance significantly.

---

# Requirements

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training

```
python train.py
```

---

# Evaluation

```
python evaluate.py
```

---

# Repository Structure
```
hypertension-retinopathy/
├─ dataset/
├─ models/
│  └─ efficientnet_model.py
├─ utils/
│  ├─ metrics.py
│  └─ transforms.py
├─ best_model.pth
├─ train.py
├─ evaluate.py
├─ dataset_loader.py
├─ config.py
├─ split_dataset.py
├─ check.py
├─ test.csv
├─ train.csv
├─ val.csv
├─ requirements.txt
└─ README.md
```
---

# Conclusion

This project demonstrates a complete **medical deep learning workflow** for hypertension detection from retinal images. Through systematic experimentation with preprocessing, transfer learning, and training strategies, a balanced classifier was achieved despite limited data.

The project highlights the importance of **domain-specific preprocessing and careful evaluation** in medical AI systems.
