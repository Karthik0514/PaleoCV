import torch
import torch.nn.functional as F
import cv2

from torchvision import transforms

from model import build_model
import config


# ==========================================
# LOAD MODEL
# ==========================================

checkpoint = torch.load(
    "best_model.pth",
    map_location=config.DEVICE
)

label_map = checkpoint["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}

model = build_model(len(label_map)).to(config.DEVICE)

model.load_state_dict(checkpoint["model"])
model.eval()


# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor()
])


# ==========================================
# INFERENCE FUNCTION
# ==========================================


def classify_fossil(img_path):

    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():

        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k=3)

    predictions = []

    for idx, score in zip(topk.indices, topk.values):

        predictions.append({
            "class": inv_label_map[idx.item()],
            "confidence": float(score.item() * 100)
        })

    return predictions

