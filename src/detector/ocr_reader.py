import easyocr
import cv2

reader = easyocr.Reader(['pl'], gpu=False)

def ocr_plate(plate_img):
    # EasyOCR expects RGB, ale mamy grayscale – działa OK, ale można powtórzyć 3 kanały
    if len(plate_img.shape) == 2:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)
    result = reader.readtext(plate_img, allowlist='QWERTYUIOPASDFGHJKLZXCVBNM1234567890')
    if result:
        return result[0][1].replace(" ", "").replace("-", "")
    return ""
