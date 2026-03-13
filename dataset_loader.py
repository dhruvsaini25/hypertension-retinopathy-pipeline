import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


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

        # extract green channel (best for retinal vessels)
        green = image[:,:,1]

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        green = clahe.apply(green)

        # convert back to 3 channel
        image = cv2.merge([green, green, green])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label