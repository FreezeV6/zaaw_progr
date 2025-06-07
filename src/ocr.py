import pytesseract
import cv2
from config import TESSERACT_CMD
import os
import imutils
import re
import numpy as np

_PATTERNS = [
    # 1) Standard plates with two-letter prefixes
    re.compile(r"^[A-Z]{2}\d{5}$"),
    re.compile(r"^[A-Z]{2}\d{4}[A-Z]$"),
    re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$"),
    re.compile(r"^[A-Z]{2}\d[A-Z]\d{3}$"),
    re.compile(r"^[A-Z]{2}\d[A-Z]{2}\d{2}$"),
    # 2) Standard plates with three-letter prefixes
    re.compile(r"^[A-Z]{3}\d{5}$"),
    re.compile(r"^[A-Z]{3}\d{4}[A-Z]$"),
    re.compile(r"^[A-Z]{3}\d{3}[A-Z]{2}$"),
    re.compile(r"^[A-Z]{3}[A-Z]\d{3}$"),
    # 3) Motorcycle / moped plates (5–6 characters)
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d?$"),
    re.compile(r"^[A-Z]{2}\d{3}$"),
    re.compile(r"^[A-Z]{3}\d{2}$"),
    re.compile(r"^[A-Z]{3}\d[A-Z]$"),
    # 4) Historic yellow plates (5 characters)
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]$"),
    re.compile(r"^[A-Z]{2}\d{3}$"),
    re.compile(r"^[A-Z]{3}\d{2}$"),
    re.compile(r"^[A-Z]{3}[A-Z]\d$"),
    # 5) Custom plates (5–7 characters, letter-digit prefix)
    re.compile(r"^[A-Z]\d[A-Z]{3,5}$"),
    re.compile(r"^[A-Z]\d[A-Z]{1,3}\d{1,2}$"),
    # 6) Diplomatic plates (W + 6 digits)
    re.compile(r"^W\d{6}$"),
    # 7) Military plates
    re.compile(r"^U[A-Z]\d{5}$"),
    re.compile(r"^U[A-Z]\d{4}T$"),
    # 8) Professional plates (1 letter + 4 digits + P + 2 chars)
    re.compile(r"^[A-Z]\d{4}P(?:\d{2}|\d[A-Z])$"),
    # 9) Reduced size ("USA") plates
    re.compile(r"^[A-Z]\d{3}$"),
    re.compile(r"^[A-Z]\d{2}[A-Z]$"),
    re.compile(r"^[A-Z]\d[A-Z]\d$"),
    re.compile(r"^[A-Z][A-Z]\d{2}$"),
    re.compile(r"^[A-Z]{3}\d$"),
    # 10) Service vehicles (H + service letter + region letter + 3 digits)
    re.compile(r"^H[ABCKMPW]\w\d{3}$"),
]


def validate_plate(text: str) -> bool:
    """Return True if ``text`` matches any known license plate format."""
    return any(p.fullmatch(text) for p in _PATTERNS)

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
    if avg_conf < ocr_conf_min or not validate_plate(text):
        return ""
    return text