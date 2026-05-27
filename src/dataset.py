import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class FossilDataset(Dataset):

    def __init__(self, csv_file, label_map, transform=None):
        self.df = pd.read_csv(csv_file)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = row["file_path"]
        label_name = row["label"]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.label_map[label_name]

        return image, label
