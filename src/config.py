import torch

DATA_CSV = "../data/annotations/labels.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4

NUM_CLASSES = 4   # change to your number of classes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
