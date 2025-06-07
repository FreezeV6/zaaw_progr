import pytesseract
import cv2
from config import TESSERACT_CMD
import os
import imutils
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def deskew(image: np.ndarray, apply: bool = True, border_mode: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Rotate image to correct minor perspective/rotation issues."""
    if not apply:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=border_mode)
    return rotated


def preprocess_plate(
    plate_img,
    width: int = 500,
    bilateral_d: int = 11,
    sigma_color: int = 41,
    sigma_space: int = 21,
    block_size: int = 25,
    c: int = 15,
    thresh_method: str = "gaussian",
    inv: bool = True,
    deskew_apply: bool = True,
    deskew_border: int = cv2.BORDER_REPLICATE,
):
    """Apply deskew and thresholding to prepare image for OCR."""
    plate_img = deskew(plate_img, apply=deskew_apply, border_mode=deskew_border)
    img = imutils.resize(plate_img, width=width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, bilateral_d, sigma_color, sigma_space)
    if thresh_method == "gaussian":
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if inv else None,
            block_size,
            c,
        )
    elif thresh_method == "mean":
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if inv else None,
            block_size,
            c,
        )
    else:  # otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV if inv else 1 + cv2.THRESH_OTSU)
    return thresh

def recognize_plate(plate_img, fname, preprocess_params: dict | None = None, tesseract_config: str | None = None):
    if preprocess_params is None:
        preprocess_params = {}
    if tesseract_config is None:
        tesseract_config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    img_prep = preprocess_plate(plate_img, **preprocess_params)
    text = pytesseract.image_to_string(
        img_prep,
        config=tesseract_config,
    )
    cv2.imwrite(os.path.join("test", f'plate_prep_{fname}.jpg'), img_prep)
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text