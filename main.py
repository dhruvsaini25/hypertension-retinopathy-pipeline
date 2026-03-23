import os
import cv2
import torch
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# -----------------------------
# CONFIG
# -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4

NUM_CLASSES = 2

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"

IMAGE_DIR = "dataset/1-Hypertensive Classification/1-Hypertensive Classification/1-images/1-Training Set"


# -----------------------------
# TRANSFORMS
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# -----------------------------
# PREPROCESSING
# -----------------------------

def crop_fundus(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return img[y:y+h, x:x+w]


# -----------------------------
# DATASET
# -----------------------------

class RetinaDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        print("CSV Columns:", self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_name = self.df["Image"][idx]
        label = self.df["Hypertensive"][idx]

        img_path = os.path.join(self.image_dir, img_name)

        # Safety check
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} not found!")

        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # DEBUG (only first few samples)
        if idx < 3:
            print(f"[DEBUG] Image: {img_name}, Label: {label}")

        # Crop fundus
        image = crop_fundus(image)

        # Green channel
        green = image[:, :, 1]

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green = clahe.apply(green)

        # Convert back to RGB
        image = cv2.merge([green, green, green])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# -----------------------------
# MODEL
# -----------------------------

def get_model(num_classes):

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# -----------------------------
# METRICS
# -----------------------------

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1


# -----------------------------
# TRAIN FUNCTION
# -----------------------------

def train():

    train_dataset = RetinaDataset(TRAIN_CSV, IMAGE_DIR, transform=train_transform)
    val_dataset = RetinaDataset(VAL_CSV, IMAGE_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = get_model(NUM_CLASSES).to(DEVICE)

    # Slightly reduced weight to control false positives
    weights = torch.tensor([1.0, 1.3]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()
        total_train_loss = 0

        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # VALIDATION
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model improved. Saved!\n")


# -----------------------------
# EVALUATION
# -----------------------------

def evaluate():

    dataset = RetinaDataset(TEST_CSV, IMAGE_DIR, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

    model.to(DEVICE)
    model.eval()

    preds = []
    labels_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(pred)
            labels_list.extend(labels.numpy())

    accuracy, f1 = compute_metrics(preds, labels_list)
    precision = precision_score(labels_list, preds)
    recall = recall_score(labels_list, preds)
    cm = confusion_matrix(labels_list, preds)

    print("\nEvaluation Results")
    print("-------------------")

    print("Accuracy :", accuracy)
    print("F1 Score :", f1)
    print("Precision:", precision)
    print("Recall   :", recall)

    print("\nConfusion Matrix")
    print(cm)

    print("\nMatrix Format:")
    print("[[TN FP]")
    print(" [FN TP]]")


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    print("1 → Train")
    print("2 → Evaluate")

    choice = input("Enter choice: ")

    if choice == "1":
        train()
    elif choice == "2":
        evaluate()
    else:
        print("Invalid choice")