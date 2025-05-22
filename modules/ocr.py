import cv2
import pytesseract
from PIL import Image

def recognize_plate_text(image_path, bbox):
    """
    Rozpoznaje tekst tablicy rejestracyjnej na obrazie.
    :param image_path: ścieżka do pliku obrazka.
    :param bbox: krotka (x_min, y_min, x_max, y_max) - współrzędne wykrytej tablicy.
    :return: ciąg znaków odczytanych z tablicy (string).
    """
    # Wczytaj obraz w kolorze
    img = cv2.imread(image_path)
    if img is None:
        return ""
    x_min, y_min, x_max, y_max = bbox
    # Upewnij się że współrzędne mieszczą się w granicach obrazu
    h, w = img.shape[:2]
    x_min = max(0, x_min); y_min = max(0, y_min)
    x_max = min(w, x_max); y_max = min(h, y_max)
    # Wytnij fragment obrazu z tablicą rejestracyjną
    plate_region = img[y_min:y_max, x_min:x_max]
    # Konwersja do odcieni szarości
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    # Binaryzacja (Otsu) dla polepszenia kontrastu znaków
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Konwersja na obraz PIL (pytesseract może przyjąć obraz PIL lub ścieżkę)
    pil_img = Image.fromarray(thresh)
    # Konfiguracja Tesseract: tylko litery i cyfry, pojedyncza linia
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(pil_img, config=config)
    # Oczyszczenie wyniku OCR: usunięcie zbędnych znaków białych i nowych linii
    text = text.strip()
    # Upewnij się, że wynik jest zapisany wielkimi literami (dla porównania z ground truth)
    text = text.upper()
    return text
