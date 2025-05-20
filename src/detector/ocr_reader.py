import cv2
import numpy as np
import pytesseract

def deskew(image):
    # Znajdź kontur największego prostokąta (tablica)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def preprocess_plate_for_ocr(plate_img):
    # Deskew
    img = deskew(plate_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    # Wyostrzanie
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharp = cv2.filter2D(thresh, -1, kernel)
    return sharp

def segment_characters(thresh_img):
    # Wydzielamy litery/cyfry
    chars = []
    h, w = thresh_img.shape
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []
    for c in contours:
        (x, y, cw, ch) = cv2.boundingRect(c)
        if 0.35*h < ch < 0.95*h and 0.03*w < cw < 0.25*w:  # tylko znaki o sensownym rozmiarze
            char_regions.append((x, y, cw, ch))
    # Sortujemy po x
    char_regions = sorted(char_regions, key=lambda item: item[0])
    for (x, y, cw, ch) in char_regions:
        char_img = thresh_img[y:y+ch, x:x+cw]
        char_img = cv2.resize(char_img, (34, 64))
        chars.append(char_img)
    return chars

def ocr_single_character(char_img):
    config = "--psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = pytesseract.image_to_string(char_img, config=config)
    text = ''.join([c for c in text if c.isalnum()])
    return text

def ocr_plate(plate_img):
    preprocessed = preprocess_plate_for_ocr(plate_img)
    chars = segment_characters(preprocessed)
    if len(chars) == 0:
        # fallback - cała tablica
        config = "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7"
        return pytesseract.image_to_string(preprocessed, config=config).replace(" ", "").replace("\n", "")
    recognized = ""
    for char_img in chars:
        recognized += ocr_single_character(char_img)
    return recognized
