# -*- coding: utf-8 -*-
"""
Raspberry Pi ANPR + Barrier Control
- Wykrywa i odczytuje tablice rejestracyjne
- Mierzy dokładność i czas przetwarzania 100 klatek
- Oblicza ocenę końcową zgodnie z funkcją oceny
- Jeśli ocena >= ustalony próg, podnosi szlaban (serwo)
"""
import time
import threading
import cv2
import numpy as np
import pytesseract
import re
# import RPi.GPIO as GPIO
from ultralytics import YOLO

# ---------- Konfiguracja ----------
WEIGHTS        = 'runs/detect/train/weights/best.pt'
CONF_THRESHOLD = 0.15
TESSERACT_CMD   = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
WHITELIST      = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
CAMERA_INDEX   = 0
# SERVO_PIN     = 17      # BCM pin
# OPEN_ANGLE    = 90
# CLOSE_ANGLE   = 0
# OPEN_DURATION = 3
MIN_ACCURACY   = 60
MAX_TOTAL_TIME = 60
WEIGHT_ACCURACY = 0.7
WEIGHT_TIME     = 0.3
MAX_FRAMES      = 1000

# ---------- Funkcje pomocnicze ----------
def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < MIN_ACCURACY or processing_time_sec > MAX_TOTAL_TIME:
        return 2.0
    accuracy_norm = (accuracy_percent - MIN_ACCURACY) / (100 - MIN_ACCURACY)
    time_norm     = (MAX_TOTAL_TIME - processing_time_sec) / (MAX_TOTAL_TIME - 10)
    score         = WEIGHT_ACCURACY * accuracy_norm + WEIGHT_TIME * time_norm
    grade         = 2.0 + 3.0 * score
    return round(grade * 2) / 2

# def setup_servo():
#     GPIO.setmode(GPIO.BCM)
#     GPIO.setup(SERVO_PIN, GPIO.OUT)
#     pwm = GPIO.PWM(SERVO_PIN, 50)
#     pwm.start(0)
#     return pwm

# def set_servo_angle(pwm, angle: float):
#     duty = angle / 18.0 + 2.0
#     pwm.ChangeDutyCycle(duty)
#     time.sleep(0.5)
#     pwm.ChangeDutyCycle(0)

# def open_barrier(pwm):
#     set_servo_angle(pwm, OPEN_ANGLE)
#     time.sleep(OPEN_DURATION)
#     set_servo_angle(pwm, CLOSE_ANGLE)
#     print('OTWARTE')


def order_points(pts):
    """
    Posortowanie punktów prostokąta (4 punkty) w kolejności:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: najmniejsza suma współrzędnych
    rect[2] = pts[np.argmax(s)]   # bottom-right: największa suma
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: najmniejsza różnica
    rect[3] = pts[np.argmax(diff)]  # bottom-left: największa różnica
    return rect


def crop_and_deskew_plate(image, box, shrink_ratio=0.3):
    """
    Crop a tighter region around the box i wykonuje deskewing przez transformację perspektywiczną.
    shrink_ratio > 0 powoduje „ściślejsze” przycięcie.

    :param image: oryginalny obraz (BGR)
    :param box: lista [x1, y1, x2, y2]
    :param shrink_ratio: jak mocno zmniejszyć obszar (np. 0.3 oznacza, że przyciemy o 30%)
    :return: wyprostowany fragment tablicy (BGR)
    """
    x1, y1, x2, y2 = box
    h_img, w_img = image.shape[:2]
    w_box, h_box = x2 - x1, y2 - y1

    # oblicz paddingi
    pad_w = int(shrink_ratio * w_box)
    pad_h = int(shrink_ratio * h_box)
    # nowe, przycięte współrzędne (zabezpieczone przed wyjściem poza obraz)
    x1p = min(max(0, x1 + pad_w), w_img)
    y1p = min(max(0, y1 + pad_h), h_img)
    x2p = max(min(w_img, x2 - pad_w), 0)
    y2p = max(min(h_img, y2 - pad_h), 0)

    plate = image[y1p:y2p, x1p:x2p].copy()
    if plate.size == 0:
        # jeżeli przycięcie dało pusty fragment, zwracamy oryginał tej części
        return image[y1:y2, x1:x2].copy()

    # Konwersja do skali szarości i binaryzacja (Otsu) w celu wykrycia obrysu
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate  # fallback, gdy brak konturów

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    pts = cv2.boxPoints(rect).astype("float32")

    # uporządkowanie punktów i obliczenie docelowych wymiarów
    src = order_points(pts)
    (tl, tr, br, bl) = src
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, M, (maxW, maxH))
    return warped


def preprocess_plate(
    plate_img,
    # --- parametry CLAHE ---
    clahe_clip=3.0,                    # granica przycięcia L wartość histogramu
    clahe_tile_grid_size=(8, 8),       # rozmiar siatki dla CLAHE
    # --- korekcja gamma ---
    gamma=1.2,                         # wartość gamma >1 rozjaśnia obraz
    # --- wybór rodzaju rozmycia ---
    blur_method='bilateral',           # 'gaussian' lub 'bilateral'
    gaussian_kernel=(3, 3),            # rozmiar jądra dla GaussianBlur
    bilateral_params=(11, 17, 17),     # (d, sigmaColor, sigmaSpace) dla bilateralFilter
    # --- thresholding ---
    threshold_type='otsu',             # 'otsu' lub 'adaptive'
    # jeśli adaptive, poniższe parametry:
    adaptive_block_size=21,            # tylko nieparzyste, np. 15
    adaptive_C=3,
    # --- operacje morfologiczne ---
    morph_kernel_size=(3, 3),          # rozmiar jądra dla morphologia
    morph_iterations=1,
    # --- skalowanie dla OCR ---
    resize_scale=2                     # mnożnik powiększenia (np. 2x)
):
    """
    Przekształca fragment tablicy do postaci gotowej do OCR:
      1. Konwersja do skali szarości
      2. CLAHE
      3. Korekcja gamma
      4. Rozmycie (Gaussian lub Bilateral)
      5. Binaryzacja (Otsu lub Adaptive)
      6. Operacje morfologiczne (otwarcie + zamknięcie)
      7. Powiększenie (resize) do lepszego odczytu OCR

    Zwraca obraz w skali szarości (Binary) gotowy do przekazania do pytesseract.
    """
    # 1. skala szarości
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile_grid_size)
    cl = clahe.apply(gray)

    # 3. korekcja gamma
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(cl, table)

    # 4. rozmycie
    if blur_method == 'gaussian':
        blur = cv2.GaussianBlur(gamma_corrected, gaussian_kernel, 0)
    elif blur_method == 'bilateral':
        d, sigmaColor, sigmaSpace = bilateral_params
        blur = cv2.bilateralFilter(gamma_corrected, d, sigmaColor, sigmaSpace)
    else:
        raise ValueError("blur_method musi być 'gaussian' lub 'bilateral'")

    # 5. binaryzacja
    if threshold_type == 'otsu':
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_type == 'adaptive':
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            adaptive_block_size, adaptive_C
        )
    else:
        raise ValueError("threshold_type musi być 'otsu' lub 'adaptive'")

    # 6. operacje morfologiczne
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # 7. powiększenie obrazu (rozciągnięcie)
    final = cv2.resize(
        closed,
        None,
        fx=resize_scale,
        fy=resize_scale,
        interpolation=cv2.INTER_CUBIC
    )

    return final


# ---------- Główna aplikacja ----------
if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    model = YOLO(WEIGHTS)
    # servo = setup_servo()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        exit(1)

    frame_count = 0
    ocr_success = 0
    start_time  = time.time()
    i=0

    print("Start ANPR + Barrier Control")
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("Błąd odczytu klatki z kamery")
            break
        frame_count += 1

        result = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)[0]
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            proc = preprocess_plate(roi)
            cv2.imwrite(f'testing/plate{i+1}', proc)
            config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={WHITELIST}"
            text = pytesseract.image_to_string(proc, config=config)
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if clean:
                ocr_success += 1
                # threading.Thread(target=open_barrier, args=(servo,), daemon=True).start()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,  # obraz
                    clean,  # tekst do wyświetlenia
                    (x1, y1 - 10),  # lewy górny róg tekstu
                    cv2.FONT_HERSHEY_SIMPLEX,  # czcionka
                    1.0,  # rozmiar czcionki
                    (0, 255, 0),  # kolor (B, G, R)
                    2  # grubość linii
                )
                break
        cv2.imshow('Live Plate Detection', frame)

        # Obsługa okna i możliwość przerwania klawiszem 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_time = time.time() - start_time
    accuracy   = (ocr_success / frame_count) * 100 if frame_count else 0
    grade      = calculate_final_grade(accuracy, total_time)

    print(f"Processed frames: {frame_count}")
    print(f"OCR Accuracy: {accuracy:.2f}%")
    print(f"Total time ({frame_count} frames): {total_time:.2f}s")
    print(f"Final grade: {grade}")

    cap.release()
    cv2.destroyAllWindows()
