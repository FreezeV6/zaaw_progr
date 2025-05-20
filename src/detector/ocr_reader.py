import pytesseract
import cv2
import numpy as np

def ocr_plate(img):

    # Wypróbuj kilka wersji, wybierz najlepszą
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpened = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
    norm = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    enhanced = cv2.addWeighted(norm, 1.5, blur, -0.5, 0)

    images = [gray, sharpened, norm, enhanced, img]

    best_text = ""
    for im in images:
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(im, config=config)
        text = text.replace(" ", "").replace("\n", "")
        # Z polskich tablic: 5-8 znaków, pierwsze litery, potem cyfry
        if len(text) > len(best_text) and 5 <= len(text) <= 8:
            best_text = text

    return best_text


