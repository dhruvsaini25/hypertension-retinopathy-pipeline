import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset_loader import RetinaDataset
from models.efficientnet_model import get_model
from utils.transforms import train_transform, val_transform
from config import *

# -----------------------------
# DATASETS
# -----------------------------

train_dataset = RetinaDataset(
    TRAIN_CSV,
    IMAGE_DIR,
    transform=train_transform
)

val_dataset = RetinaDataset(
    VAL_CSV,
    IMAGE_DIR,
    transform=val_transform
)

# -----------------------------
# DATALOADERS
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# MODEL
# -----------------------------

model = get_model(NUM_CLASSES_HYPERTENSION)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

# -----------------------------
# TRAINING LOOP
# -----------------------------

for epoch in range(EPOCHS):

    # ---- TRAIN ----
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

    # ---- VALIDATION ----
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

    # ---- SAVE BEST MODEL ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model improved. Saved!\n")