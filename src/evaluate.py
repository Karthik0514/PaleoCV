import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from dataset import FossilDataset
from model import build_model
import config

def main():

    checkpoint = torch.load("best_model.pth", map_location=config.DEVICE)

    label_map = checkpoint["label_map"]
    inv_map = {v:k for k,v in label_map.items()}

    df = pd.read_csv(config.DATA_CSV)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    ds = FossilDataset(config.DATA_CSV, label_map, tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    model = build_model(len(label_map))

    model.load_state_dict(checkpoint["model"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)

            out = model(x)
            p = out.argmax(1).cpu().numpy()

            y_pred.extend(p)
            y_true.extend(y.numpy())

    print(classification_report(
        y_true,
        y_pred,
        target_names=[inv_map[i] for i in range(len(inv_map))]
    ))

    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
