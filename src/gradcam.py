import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

from model import build_model
import config


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)

        score = out[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()

        cam = cv2.resize(cam, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


def occlusion_test(model, img, label_idx):
    """Occlude central region and test confidence drop"""
    h, w, _ = img.shape
    occluded = img.copy()

    # black square in center
    occluded[h//3:h//2, w//3:w//2] = 0

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    x_occ = tf(occluded).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(x_occ), dim=1)[0]
        occ_conf = probs[label_idx].item()

    return occ_conf

def generate_gradcam_from_path(img_path):

    checkpoint = torch.load(
        "best_model.pth",
        map_location=config.DEVICE
    )

    label_map = checkpoint["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    model = build_model(len(label_map)).to(config.DEVICE)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    target_layer = model.layer4[-1].conv2

    cam_generator = GradCAM(model, target_layer)

    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    x = tf(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        pred = model(x).argmax(1).item()

    cam = cam_generator.generate(x, pred)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        0.5 * heatmap +
        0.5 * cv2.resize(
            img,
            (config.IMAGE_SIZE, config.IMAGE_SIZE)
        )
    )

    overlay = overlay.astype(np.uint8)

    save_path = "outputs/gradcam_overlay.png"

    cv2.imwrite(
        save_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    return save_path


def main():

    # ===== Load model =====
    checkpoint = torch.load("best_model.pth", map_location=config.DEVICE)
    label_map = checkpoint["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    model = build_model(len(label_map)).to(config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Target layer for Grad-CAM
    target_layer = model.layer4[-1].conv2
    cam_generator = GradCAM(model, target_layer)

    
    img_path = "C:/College Stuff/paleo-cv/data/raw/Ammonites/Ammonite2.jpg"

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    x = tf(img).unsqueeze(0).to(config.DEVICE)

    # ===== Prediction + confidence =====
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k=3)

    print("\nTop-3 Predictions:")
    for idx, score in zip(topk.indices, topk.values):
        print(f"{inv_label_map[idx.item()]}: {score.item()*100:.2f}%")

    pred_class = topk.indices[0].item()
    base_conf = probs[pred_class].item()

    # ===== Grad-CAM =====
    cam = cam_generator.generate(x, pred_class)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    resized_img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    overlay = 0.5 * heatmap + 0.5 * resized_img
    overlay = overlay.astype(np.uint8)

    # ===== Save outputs =====
    cv2.imwrite("gradcam_original.png",
                cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

    cv2.imwrite("gradcam_raw.png", np.uint8(255 * cam))

    cv2.imwrite("gradcam_overlay.png",
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("\nSaved:")
    print(" - gradcam_original.png")
    print(" - gradcam_raw.png")
    print(" - gradcam_overlay.png")

    # ===== Occlusion sensitivity =====
    occ_conf = occlusion_test(model, img, pred_class)

    print(f"\nConfidence (original): {base_conf*100:.2f}%")
    print(f"Confidence (occluded): {occ_conf*100:.2f}%")

    if occ_conf < base_conf:
        print("→ Model relies on the occluded region (important visual cue).")
    else:
        print("→ Occlusion had limited effect.")


if __name__ == "__main__":
    main()
