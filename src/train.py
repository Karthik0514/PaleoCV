import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import FossilDataset
from model import build_model
import config

def main():

    df = pd.read_csv(config.DATA_CSV)

    class_names = sorted(df["label"].unique())
    label_map = {c: i for i, c in enumerate(class_names)}

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_csv = "train_tmp.csv"
    val_csv = "val_tmp.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_ds = FossilDataset(train_csv, label_map, train_tf)
    val_ds = FossilDataset(val_csv, label_map, val_tf)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_model(len(label_map))


    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = 0

    for epoch in range(config.NUM_EPOCHS):

        model.train()
        train_loss = 0

        for x, y in tqdm(train_loader):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            optimizer.zero_grad()

            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                out = model(x)
                pred = out.argmax(1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} | Train loss {train_loss:.4f} | Val acc {acc:.4f}")

        if acc > best_val:
            best_val = acc
            torch.save({
                "model": model.state_dict(),
                "label_map": label_map
            }, "best_model.pth")

if __name__ == "__main__":
    main()
