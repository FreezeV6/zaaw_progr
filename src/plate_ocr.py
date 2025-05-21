import easyocr
import cv2

class PlateOCR:
    def __init__(self, gpu: bool = False):
        # tylko znaki A-Z i 0-9
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    def read_plate(self, image):
        """
        image: ROI tablicy (BGR lub gray). Zwraca (text, confidence).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        # wybieramy najwyższą pewność
        best = max(results, key=lambda x: x[2])
        text = best[1].replace(" ", "")
        return text, best[2]
