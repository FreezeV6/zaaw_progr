import time
from utils import crop_image, compute_iou

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    acc_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 60
    score = 0.7 * acc_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2

def evaluate(detector, ocr, records, images_dir):
    """
    Przetwarza listę rekordów (z kluczem 'filename','bbox','text'):
    - detekcja
    - crop + OCR
    - liczy accuracy, średnie IoU, czas
    - zwraca accuracy%, mean_iou, total_time, final_grade
    """
    correct = 0
    ious = []
    start = time.time()

    for rec in records:
        import cv2
        img = cv2.imread(rec['filename'])
        # ground-truth
        true_box = rec['bbox']
        true_text = rec['text']
        # detekcja
        dets = detector.detect(img)
        if not dets:
            ious.append(0)
            continue
        # wybieramy najwyższe conf
        det = max(dets, key=lambda x: x[4])
        pred_box = det[:4]
        ious.append(compute_iou(pred_box, true_box))
        # OCR
        crop = crop_image(img, pred_box)
        pred_text, _ = ocr.read_plate(crop)
        if pred_text.upper() == true_text.upper():
            correct += 1

    total_time = time.time() - start
    accuracy = 100 * correct / len(records)
    mean_iou  = sum(ious) / len(ious) * 100
    grade = calculate_final_grade(accuracy, total_time)
    return accuracy, mean_iou, total_time, grade
