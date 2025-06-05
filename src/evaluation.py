import time
from utils import iou, crop_bbox
from ocr import recognize_plate
import cv2
import os

def evaluate(detector, data, images_dir):
    total, correct, total_iou = 0, 0, []
    start = time.time()
    for row in data[:100]:
        fname, xtl, ytl, xbr, ybr, gt_plate = row
        img_path = f"{images_dir}/{fname}"
        img = cv2.imread(img_path)
        if img is None:
            continue
        dets = detector.detect(img)
        if len(dets) == 0:
            continue
        best = dets[0][:4]
        total_iou.append(iou([float(xtl), float(ytl), float(xbr), float(ybr)], best))
        plate_img = crop_bbox(img, best)
        cv2.imwrite(os.path.join("test", f'plate_{fname}'), plate_img)
        pred = recognize_plate(plate_img, fname)
        print(f'{fname}, {gt_plate}, {pred}')
        if pred == gt_plate:
            correct += 1
        total += 1
    elapsed = time.time() - start
    accuracy = correct / total if total else 0
    avg_iou = sum(total_iou) / len(total_iou) if total_iou else 0
    return accuracy, elapsed, avg_iou

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2
