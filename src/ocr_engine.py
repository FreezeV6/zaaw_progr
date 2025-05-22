import cv2
import pytesseract
import easyocr

class OCREngine:
    def __init__(self, use_easyocr: bool = False):
        self.use_easyocr = use_easyocr
        if self.use_easyocr:
            # English uppercase + digits
            self.reader = easyocr.Reader(['en'], gpu=True)

    def recognize(self, plate_img):
        # convert to gray
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        if self.use_easyocr:
            res = self.reader.readtext(gray, detail=0)
            # concatenate lines
            return ''.join(res).replace(' ', '').upper()
        else:
            # simple preprocessing
            # You can experiment with thresholding here
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            txt = pytesseract.image_to_string(gray, config=config)
            return txt.strip().replace(' ', '').upper()
