import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
EPOCHS = 10
BATCH_SIZE = 8
LR = 3e-4
NUM_CLASSES_HYPERTENSION = 2
NUM_CLASSES_RETINOPATHY = 4

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"

IMAGE_DIR = "dataset/1-Hypertensive Classification/1-Hypertensive Classification/1-images/1-Training Set"

TRAIN_IMAGES = "dataset/1-Hypertensive Classification/1-Hypertensive Classification/1-images/1-Training Set"