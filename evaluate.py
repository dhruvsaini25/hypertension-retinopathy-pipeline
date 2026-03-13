import torch
from torch.utils.data import DataLoader
from dataset_loader import RetinaDataset
from utils.transforms import val_transform
from utils.metrics import compute_metrics
from models.efficientnet_model import get_model
from config import *

dataset = RetinaDataset(
    TEST_CSV,
    IMAGE_DIR,
    transform=val_transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE)

model = get_model(NUM_CLASSES_HYPERTENSION)
model.load_state_dict(torch.load("best_model.pth"))

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

acc, f1 = compute_metrics(preds, labels_list)

print("Accuracy:", acc)
print("F1 Score:", f1)