import easyocr
from src.utils.config import OCR_LANGS

reader = easyocr.Reader(OCR_LANGS)

def ocr_plate(plate_img):
    result = reader.readtext(plate_img)
    if not result:
        return ""
    return max(result, key=lambda x: x[2])[1].replace(" ", "").upper()
