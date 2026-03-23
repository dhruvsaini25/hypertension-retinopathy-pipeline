import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


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


class RetinaDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        print("CSV Columns:", self.df.columns)  # debug once

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_name = self.df["Image"][idx]
        label = self.df["Hypertensive"][idx]

        img_path = os.path.join(self.image_dir, img_name)

        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} not found!")

        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        # 1️⃣ Crop fundus
        image = crop_fundus(image)

        # 2️⃣ Green channel
        green = image[:, :, 1]

        # 3️⃣ CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        green = clahe.apply(green)

        # 4️⃣ Back to 3 channel
        image = cv2.merge([green, green, green])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 5️⃣ PIL
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label