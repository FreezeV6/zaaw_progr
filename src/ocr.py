import pytesseract
import cv2
from config import TESSERACT_CMD, REV_CHARS_MAP, CHARS_MAP
import os
import imutils
import re
import numpy as np


PLATE_REGEX = re.compile(r"^[A-Z0-9]{4,8}$")

def validate_plate(text: str) -> bool:
    """Return True if text looks like a license plate."""
    return bool(PLATE_REGEX.fullmatch(text))

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
    blur_method: str = "bilateral",
    gaussian_kernel: tuple[int, int] = (5, 5),
    bilateral_d: int = 11,
    sigma_color: int = 41,
    sigma_space: int = 21,
    gamma: float = 1.0,
    clahe_clip: float | None = None,
    clahe_tile_grid: tuple[int, int] | None = None,
    block_size: int = 25,
    adaptive_block: int = 25,
    adaptive_C: int = 0,
    c: int = 15,
    thresh_method: str = "gaussian",
    inv: bool = True,
    deskew_apply: bool = True,
    deskew_border: int = cv2.BORDER_REPLICATE,
    kernel_size: int = 3,
    open_iter: int = 0,
    close_iter: int = 0,
    dilate_iter: int = 0,
):
    """Apply deskew and thresholding to prepare image for OCR."""
    plate_img = deskew(plate_img, apply=deskew_apply, border_mode=deskew_border)
    img = imutils.resize(plate_img, width=width)
    if gamma != 1.0:
        inv_gamma = 1.0 / max(gamma, 1e-8)
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe_clip is not None and clahe_tile_grid is not None:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile_grid)
        gray = clahe.apply(gray)
    if blur_method == "gaussian":
        gray = cv2.GaussianBlur(gray, gaussian_kernel, 0)
    else:
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
    elif thresh_method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if inv else None,
            adaptive_block,
            adaptive_C,
        )
    else:  # otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV if inv else 1 + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if open_iter > 0:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if dilate_iter > 0:
        thresh = cv2.dilate(thresh, kernel, iterations=dilate_iter)
    return thresh

def replace_chars(text: str, split: int | None, rev: bool = False) -> str:
    c_map = CHARS_MAP if not rev else REV_CHARS_MAP
    if split is None:
        for c in text:
            if c not in c_map.keys():
                continue
            text = text.replace(c, c_map[c])
        return text
    for i, c in enumerate(text[:split]):
        if c in c_map.keys():
            text = "".join([text[:split].replace(c, CHARS_MAP[c]), text[split:]])
    return text

def process_text(text: str) -> str:
    if len(text) < 4:
        return ""
    if text[0] in "AIM0123456789":
        text = text[1:]
    if text[0:3] != "BI" and text[0] == "B":
        text = text[1:]
    if text[0] in CHARS_MAP.keys():
        text = "".join([CHARS_MAP[text[0]], text[1:]])
    if text[:2].isalnum() and not text[2].isalnum() and len(text) > 7:
        text = text[:8]
    elif text[:3].isalnum() and len(text) > 8:
        text = text[:8]
    if len(text) > 1 and text[0] in "AIM0123456789":
        text = text[1:]
    if len(text) > 8:
        text = text[:8]
    if len(text) > 7:
        replace_chars(text, 3)
    elif len(text) > 6:
        replace_chars(text, 2)
    if len(text) > 2 and text[:2].isalnum() and not text[2].isalnum():
        text = text[:7]
    if len(text) == 7 and text[0].isalnum() and not text[1].isalnum():
        text = "".join([text[0], CHARS_MAP[text[1]], text[2:]])
    if len(text) > 7 and text[:3].isalnum() and not text[3].isalnum():
        text = text[:8]
    if len(text) == 8:
        text = replace_chars(text, 3)
    if len(text) == 7:
        text = replace_chars(text, 2)
    if text[:2].isalnum():
        text
    if text[:2].isalnum():
        text
    return text

def recognize_plate(
    plate_img,
    fname,
    preprocess_params: dict | None = None,
    tesseract_config: str | None = None,
    ocr_conf_min: float = 0.0,
):
    if preprocess_params is None:
        preprocess_params = {}
    if tesseract_config is None:
        tesseract_config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    img_prep = preprocess_plate(plate_img, **preprocess_params)
    data = pytesseract.image_to_data(
        img_prep,
        output_type=pytesseract.Output.DICT,
        config=tesseract_config,
    )
    text = "".join(data.get("text", [])).upper()
    confs = [float(c) for c in data.get("conf", []) if c != "-1"]
    avg_conf = (sum(confs) / len(confs) / 100) if confs else 0.0
    cv2.imwrite(os.path.join("test", f"plate_prep_{fname}.jpg"), img_prep)
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    text = process_text(text)
    return text