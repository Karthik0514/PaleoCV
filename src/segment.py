import cv2
import numpy as np


def segment_fossil(img_path):

    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # invert if necessary
    white_ratio = np.sum(thresh == 255) / thresh.size

    if white_ratio > 0.5:
        thresh = 255 - thresh

    # morphology cleanup
    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise RuntimeError("No fossil contour detected")

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros(gray.shape, dtype=np.uint8)

    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    segmented = cv2.bitwise_and(original, original, mask=mask)

    return {
        "original": original,
        "mask": mask,
        "segmented": segmented,
        "contour": largest
    }
    
    if __name__ == "__main__":

        img_path = "C:/College Stuff/paleo-cv/data/raw/Ammonites/Ammonite2.jpg"

        result = segment_fossil(img_path)

        cv2.imwrite("segment_mask.png", result["mask"])
        cv2.imwrite("segment_result.png", result["segmented"])

        contour_vis = result["original"].copy()

        cv2.drawContours(
            contour_vis,
            [result["contour"]],
            -1,
            (0, 255, 0),
            2
        )
    
    
        cv2.imwrite("segment_contour.png", contour_vis)
    
        print("Saved segmentation outputs")