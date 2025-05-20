import pytesseract
from paddleocr import PaddleOCR
import logging
from src.utils.ocr_preprocess import preprocess_for_ocr

class PlateOCR:
    def __init__(self, mode='tesseract', psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        self.mode = mode
        self.psm = psm
        self.whitelist = whitelist
        if mode == 'paddle':
            self.ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)
    def recognize(self, img):
        img = preprocess_for_ocr(img)
        if self.mode == 'tesseract':
            config = f'--psm {self.psm} -c tessedit_char_whitelist={self.whitelist}'
            text = pytesseract.image_to_string(img, config=config)
            text = self.clean_plate(text)
        else:
            result = self.ocr.ocr(img, cls=True)
            text = ""
            if result and len(result[0]) > 0:
                text = result[0][0][1][0]
            text = self.clean_plate(text)
        logging.info(f"OCR: {text}")
        return text
    @staticmethod
    def clean_plate(s):
        return ''.join([c for c in s if c.isalnum()])
