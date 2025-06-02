import os
import cv2
import numpy as np
import pytesseract
import re
import time
import xml.etree.ElementTree as ET
from ultralytics import YOLO


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_and_deskew_plate(image, box, shrink_ratio=0.1):
    x1, y1, x2, y2 = box
    h_img, w_img = image.shape[:2]
    w_box, h_box = x2 - x1, y2 - y1
    # mniejszy shrink, żeby uchwycić krawędzie tablicy
    pad_w = int(shrink_ratio * w_box)
    pad_h = int(shrink_ratio * h_box)
    x1p = np.clip(x1 + pad_w, 0, w_img)
    y1p = np.clip(y1 + pad_h, 0, h_img)
    x2p = np.clip(x2 - pad_w, 0, w_img)
    y2p = np.clip(y2 - pad_h, 0, h_img)
    plate = image[y1p:y2p, x1p:x2p].copy()
    if plate.size == 0:
        return image[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    # Nieco silniejszy bilinear blur, by wygładzić tłumienie
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Odwrócenie, jeśli tło białe
    if np.mean(thr) > 127:
        thr = cv2.bitwise_not(thr)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    pts = cv2.boxPoints(rect).astype("float32")
    src = order_points(pts)
    tl, tr, br, bl = src
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))
    # minimalne wymiary, by uniknąć zbyt wąskich obrazów
    maxW = max(maxW, 100)
    maxH = max(maxH, 30)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, M, (maxW, maxH))
    # jeśli orientacja pionowa, obróć
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def preprocess_plate(
    plate_img,
    clahe_clip=3.0,
    clahe_tile_grid_size=(8, 8),
    gamma=1.2,
    blur_method='bilateral',
    gaussian_kernel=(5, 5),
    bilateral_params=(9, 75, 75),
    threshold_type='otsu',
    adaptive_block_size=25,
    adaptive_C=5,
    morph_kernel_size=(3, 3),
    morph_iterations=2,
    resize_scale=2
):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # CLAHE z wyższym clipLimit, by podbić kontrast liter
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile_grid_size)
    cl = clahe.apply(gray)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(cl, table)
    if blur_method == 'gaussian':
        blur = cv2.GaussianBlur(gamma_corrected, gaussian_kernel, 0)
    else:
        d, sigmaColor, sigmaSpace = bilateral_params
        blur = cv2.bilateralFilter(gamma_corrected, d, sigmaColor, sigmaSpace)
    if threshold_type == 'adaptive':
        # stosujemy THRESH_BINARY zamiast INV, zostawiamy białe litery na czarnym tle
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            adaptive_block_size, adaptive_C
        )
    else:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    final = cv2.resize(closed, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_CUBIC)
    return final


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2


def load_annotations(xml_path):
    gt_dict = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image_el in root.findall('image'):
        img_name = image_el.get('name')
        plate_number = None
        for box_el in image_el.findall('box'):
            for attr_el in box_el.findall('attribute'):  # atrybuty inside
                if attr_el.get('name') == 'plate number':
                    plate_number = attr_el.text.strip().upper()
                    break
            if plate_number:
                break
        if img_name and plate_number:
            gt_dict[img_name] = plate_number
    return gt_dict


def run_test(
    weights='runs/detect/train/weights/best.pt',
    test_dir='dataset/images/train',
    annotations_xml='data/annotations.xml',
    tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    model = YOLO(weights)
    gt_dict = load_annotations(annotations_xml)

    total_images = 0
    correct_count = 0
    detection_count = 0
    start_time = time.time()

    for fname in sorted(os.listdir(test_dir))[:100]:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if fname not in gt_dict:
            continue
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        total_images += 1
        gt_plate = gt_dict[fname]
        results = model.predict(source=img, conf=0.15, verbose=False)[0]
        all_boxes = [tuple(map(int, box)) for box in results.boxes.xyxy]
        if all_boxes:
            detection_count += 1
            best_box = max(all_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            print(f"{fname} - Detekcja: {len(all_boxes)} boxów, najlepszy: {best_box}")
        else:
            best_box = None
            print(f"{fname} - Brak detekcji")
        predicted_plate = ""
        if best_box:
            desk = crop_and_deskew_plate(img, list(best_box), shrink_ratio=0.1)
            proc = preprocess_plate(
                plate_img=desk,
                clahe_clip=3.0,
                clahe_tile_grid_size=(8,8),
                gamma=1.2,
                blur_method='bilateral',
                gaussian_kernel=(5,5),
                bilateral_params=(9,75,75),
                threshold_type='otsu',
                adaptive_block_size=25,
                adaptive_C=5,
                morph_kernel_size=(3,3),
                morph_iterations=2,
                resize_scale=2
            )
            os.makedirs('test', exist_ok=True)
            cv2.imwrite(os.path.join('test', f"plate_{fname}"), proc)
            config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}"
            txt = pytesseract.image_to_string(proc, config=config)
            clean = re.sub(r'[^A-Z0-9]', '', txt.upper()).strip()
            if clean:
                predicted_plate = clean
                print(f"    OCR: '{predicted_plate}' (GT: '{gt_plate}')")
            else:
                print(f"    OCR: brak odczytu (GT: '{gt_plate}')")
        if predicted_plate == gt_plate:
            correct_count += 1

    duration = time.time() - start_time
    accuracy = (correct_count / total_images * 100) if total_images else 0
    detection_rate = (detection_count / total_images * 100) if total_images else 0
    grade = calculate_final_grade(accuracy, duration)

    print("\n=== Podsumowanie ===")
    print(f"Przetworzone: {total_images}")
    print(f"Detekcja: {detection_count}/{total_images} = {detection_rate:.2f}%")
    print(f"OCR poprawny: {correct_count}/{total_images} = {accuracy:.2f}%")
    print(f"Czas: {duration:.2f}s")
    print(f"Ocena końcowa: {grade}")

if __name__ == '__main__':
    run_test()
