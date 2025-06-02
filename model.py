import os
import cv2
import numpy as np
import pytesseract
import re
from ultralytics import YOLO


# ---------- Pomocnicze funkcje ----------

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
    adaptive_block_size=15,            # tylko nieparzyste, np. 15
    adaptive_C=2,
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


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    """
    Oblicza końcową ocenę na podstawie:
      - accuracy_percent: dokładność OCR w %
      - processing_time_sec: czas przetwarzania 100 obrazów (w sekundach)
    Zwraca ocenę z zakresu 2.0–5.0, zaokrągloną do 0.5.
    """
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40  # 60%→0.0, 100%→1.0
    time_norm = (60 - processing_time_sec) / 50    # 60s→0.0,10s→1.0
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2


def run_model(
    weights='runs/detect/train/weights/best.pt',
    test_dir='dataset/images/train',
    # output_dir='dataset/images/test',
    output_dir='runs/detect/test_results',
    tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    # ----- dodatkowe parametry preprocesingu: -----
    shrink_ratio=0.08,
    clahe_clip=4.0,
    clahe_tile_grid_size=(8, 8),
    gamma=1.2,
    blur_method='bilateral',
    gaussian_kernel=(3, 3),
    bilateral_params=(11, 17, 17),
    threshold_type='otsu',
    adaptive_block_size=15,
    adaptive_C=2,
    morph_kernel_size=(3, 3),
    morph_iterations=1,
    resize_scale=2
):
    """
    Główna pętla:
      1. Wczytuje modele i obrazy testowe
      2. Dla każdej wykrytej tablicy:
         - crop + deskew
         - preprocess zgodnie z parametrami
         - pass to Tesseract OCR
      3. Zapisuje wyniki (zdjęcia i tekst na stdout)

    Parametry preprocesingu można sterować przez przekazanie własnych wartości
    do argumentów funkcji run_model (np. run_model(clahe_clip=4.0, blur_method='gaussian', ...)).
    """
    # wskazanie binarki Tesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    model = YOLO(weights)

    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(test_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model.predict(source=image, conf=0.15, verbose=False)[0]
        for i, box in enumerate(results.boxes.xyxy):
            box = list(map(int, box))
            deskewed = crop_and_deskew_plate(image, box, shrink_ratio=shrink_ratio)
            proc = preprocess_plate(
                plate_img=deskewed,
                clahe_clip=clahe_clip,
                clahe_tile_grid_size=clahe_tile_grid_size,
                gamma=gamma,
                blur_method=blur_method,
                gaussian_kernel=gaussian_kernel,
                bilateral_params=bilateral_params,
                threshold_type=threshold_type,
                adaptive_block_size=adaptive_block_size,
                adaptive_C=adaptive_C,
                morph_kernel_size=morph_kernel_size,
                morph_iterations=morph_iterations,
                resize_scale=resize_scale
            )

            # zapis przetworzonego fragmentu (dla debugowania/analizy)
            cv2.imwrite(os.path.join(output_dir, f"plate_{fname}"), proc)

            # wywołanie Tesseract z ograniczeniem whitelist
            config = f"--oem 3 --psm 8 -c tessedit_char_whitelist={whitelist}"
            text = pytesseract.image_to_string(proc, config=config)
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if clean:
                print(f"{fname}[{i}] OCR: {clean}")
            else:
                print(f"{fname}[{i}] No read")

        # zapis oryginalnego obrazu z ramkami detekcji
        cv2.imwrite(os.path.join(output_dir, fname), image)

    print(f"Done: results in {output_dir}")


if __name__ == '__main__':
    run_model(
        # clahe_clip=4.0,
        # blur_method='gaussian',
        # gaussian_kernel=(3, 3),
        # threshold_type='adaptive',
        # adaptive_block_size=11,
        # adaptive_C=3,
        # resize_scale=3
    )

