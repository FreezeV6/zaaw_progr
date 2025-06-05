import pytesseract
import cv2
from config import TESSERACT_CMD
import os
import imutils
import re

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def preprocess_plate(plate_img):
    img = imutils.resize(plate_img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 41, 21)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 15
    )
    return thresh

def recognize_plate(plate_img, fname):
    img_prep = preprocess_plate(plate_img)
    text = pytesseract.image_to_string(
        img_prep,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    cv2.imwrite(os.path.join("test", f'plate_prep_{fname}.jpg'), img_prep)
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text

