# src/debug_utils.py

import os, cv2, re, csv
import numpy as np
from ultralytics import YOLO
import pytesseract

def save_plate_crops(model_path: str,
                     images_dir: str,
                     output_dir: str,
                     conf_thresh: float = 0.2):
    """Wykryj tablice YOLOm i zapisz każdy crop do output_dir."""
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(('.jpg','.png')): continue
        img = cv2.imread(os.path.join(images_dir, fn))
        res = model(img, conf=conf_thresh)[0]
        for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
            x1,y1,x2,y2 = box[:4].astype(int)
            crop = img[y1:y2, x1:x2]
            outp = os.path.join(output_dir, f"{os.path.splitext(fn)[0]}_{i}.jpg")
            cv2.imwrite(outp, crop)
    print(f"[debug] saved crops in {output_dir}")

def preprocess_for_tesseract(crop: np.ndarray) -> np.ndarray:
    """Zwraca obraz binarny gotowy pod Tesseract (--psm 7)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )
    return th

def ocr_tesseract(prep_img: np.ndarray) -> str:
    """Uruchom Tesseract z whitelistą i odetnij białe znaki."""
    cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    txt = pytesseract.image_to_string(prep_img, config=cfg)
    return txt.strip()

_confusion_map = {
    'O':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6'
}
_plate_regex = re.compile(r'^[A-Z]{2}\d{5}$')

def apply_confusion_map(txt: str) -> str:
    """Zamienia np. O -> 0, S -> 5 itd."""
    return ''.join(_confusion_map.get(c, c) for c in txt)

def ocr_on_crops(crops_dir: str, output_csv: str):
    """
    Dla każdego cropa:
     - wczyta go,
     - preprocess,
     - OCR,
     - mapuje pomyłki,
     - zapisze: filename, ocr_raw, ocr_fixed, valid_format
    """
    rows = []
    for fn in os.listdir(crops_dir):
        if not fn.lower().endswith(('.jpg','.png')): continue
        img = cv2.imread(os.path.join(crops_dir, fn))
        prep = preprocess_for_tesseract(img)
        raw = ocr_tesseract(prep)
        fixed = apply_confusion_map(raw)
        valid = bool(_plate_regex.match(fixed))
        rows.append([fn, raw, fixed, valid])
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file','ocr_raw','ocr_fixed','valid_plate'])
        w.writerows(rows)
    print(f"[debug] OCR results written to {output_csv}")
