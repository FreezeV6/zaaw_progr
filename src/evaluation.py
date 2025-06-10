import time

from tqdm import tqdm

from utils import iou, crop_bbox
from ocr import recognize_plate
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import threading

def evaluate(
    detector,
    data,
    images_dir,
    num_threads: int = 8,
    preprocess_params: dict | None = None,
    crop_offsets: dict | None = None,
    conf: float | None = None,
    nms_iou: float | None = None,
    shrink_ratio: float = 0.0,
    tesseract_config: str | None = None,
    ocr_conf_min: float = 0.0,
):
    total, correct, total_iou = 0, 0, []
    start = time.time()

    if crop_offsets is None:
        crop_offsets = {"x1": 15, "x2": 0, "y1": 0, "y2": 0}
    lock = threading.Lock()

    def process(row):
        fname, xtl, ytl, xbr, ybr, gt_plate = row
        img_path = f"{images_dir}/{fname}"
        img = cv2.imread(img_path)
        if img is None:
            return None
        with lock:
            dets = detector.detect(img, conf=conf, iou=nms_iou)
        if len(dets) == 0:
            return None
        best = dets[0][:4]
        val_iou = iou([float(xtl), float(ytl), float(xbr), float(ybr)], best)
        plate_img = crop_bbox(img, best, offsets=crop_offsets, shrink_ratio=shrink_ratio)
        pred_text, _ = recognize_plate(
            plate_img,
            fname,
            preprocess_params,
            tesseract_config,
            ocr_conf_min,
        )
        # print(fname) if pred_text != gt_plate else None
        return (pred_text == gt_plate, val_iou)

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        results = list(tqdm(ex.map(process, data), total=len(data), desc="Przetwarzanie tablic"))

    for res in results:
        if res is None:
            continue
        pred_correct, val_iou = res
        if pred_correct:
            correct += 1
        total_iou.append(val_iou)
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