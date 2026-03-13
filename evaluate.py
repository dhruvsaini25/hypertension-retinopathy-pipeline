import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from dataset_loader import RetinaDataset
from utils.transforms import val_transform
from utils.metrics import compute_metrics
from models.efficientnet_model import get_model
from config import *

# -----------------------------
# LOAD TEST DATASET
# -----------------------------

dataset = RetinaDataset(
    TEST_CSV,
    IMAGE_DIR,
    transform=val_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = get_model(NUM_CLASSES_HYPERTENSION)

model.load_state_dict(
    torch.load("best_model.pth", map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# -----------------------------
# PREDICTION
# -----------------------------

preds = []
labels_list = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        preds.extend(predictions)
        labels_list.extend(labels.numpy())

# -----------------------------
# METRICS
# -----------------------------

accuracy, f1 = compute_metrics(preds, labels_list)

precision = precision_score(labels_list, preds)
recall = recall_score(labels_list, preds)

cm = confusion_matrix(labels_list, preds)

# -----------------------------
# PRINT RESULTS
# -----------------------------

print("\nEvaluation Results")
print("-------------------")

print("Accuracy :", accuracy)
print("F1 Score :", f1)
print("Precision:", precision)
print("Recall   :", recall)

print("\nConfusion Matrix")
print(cm)

print("\nMatrix Format:")
print("[[True Negatives  False Positives]")
print(" [False Negatives True Positives]]")