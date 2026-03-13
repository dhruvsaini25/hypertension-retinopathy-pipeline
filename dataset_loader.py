import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


def crop_fundus(img):
    """
    Remove black borders around fundus image
    """

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

    cropped = img[y:y+h, x:x+w]

    return cropped


class RetinaDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        img_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]

        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        # ----------------------------------
        # 1️⃣ Crop fundus region
        # ----------------------------------

        image = crop_fundus(image)

        # ----------------------------------
        # 2️⃣ Extract green channel
        # ----------------------------------

        green = image[:, :, 1]

        # ----------------------------------
        # 3️⃣ CLAHE vessel enhancement
        # ----------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        green = clahe.apply(green)

        # ----------------------------------
        # 4️⃣ Convert back to 3-channel image
        # ----------------------------------

        image = cv2.merge([green, green, green])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ----------------------------------
        # 5️⃣ Convert to PIL for transforms
        # ----------------------------------

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label