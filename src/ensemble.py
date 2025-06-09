import cv2
import imutils
import pytesseract


def process_image(img) -> str:
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 41, 21)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    plate_number = pytesseract.image_to_string(inv, config=config)
    text = plate_number.strip()
    return text