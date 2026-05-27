import cv2
import numpy as np

from segment import segment_fossil


def compute_shape_features(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / h

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area > 0 else 0

    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "circularity": circularity
    }


if __name__ == "__main__":

    img_path = "C:/College Stuff/paleo-cv/data/raw/Ammonites/Ammonite2.jpg"

    result = segment_fossil(img_path)

    features = compute_shape_features(result["contour"])

    print("\nMorphological Features")
    print("----------------------")

    for k, v in features.items():
        print(f"{k}: {v:.4f}")