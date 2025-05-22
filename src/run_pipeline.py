# run_pipeline.py

import os
import time
import cv2
import argparse

from prepare_data import DATA_DIR
from plate_detector import PlateDetector
from ocr_engine import OCREngine
from debug_utils import (
    save_plate_crops,
    preprocess_for_tesseract,
    ocr_tesseract,
    apply_confusion_map,
    ocr_on_crops
)

def load_gt(gt_path):
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth file not found at {gt_path}. Please run prepare_data.py first.")
    entries = []
    with open(gt_path, encoding='utf-8') as f:
        for line in f:
            fn, plate = line.strip().split(maxsplit=1)
            entries.append((fn, plate))
    return entries

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    # minimum requirements
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50  # 10s->1.0, 60s->0.0
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2

def main(args):
    img_dir = os.path.join(DATA_DIR, 'images')
    gt_path = os.path.join(DATA_DIR, 'labels', 'val', 'gt.txt')
    gt_entries = load_gt(gt_path)

    # initialize detector & OCR engines
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')
    detector = PlateDetector(model_path, conf_thres=0.25)
    ocr_tess = OCREngine(use_easyocr=False)
    ocr_easy = OCREngine(use_easyocr=True)

    # -- debug mode: just save crops & run batch OCR on them, then exit
    if args.debug:
        debug_dir = os.path.join(DATA_DIR, 'debug')
        crops_dir = os.path.join(debug_dir, 'crops')
        os.makedirs(crops_dir, exist_ok=True)
        print(f"[debug] Saving all detected plate crops to {crops_dir} …")
        save_plate_crops(
            model_path=model_path,
            images_dir=img_dir,
            output_dir=crops_dir,
            conf_thresh=detector.conf_thres
        )
        csv_path = os.path.join(debug_dir, 'ocr_results.csv')
        print(f"[debug] Running OCR on crops and writing results to {csv_path} …")
        ocr_on_crops(crops_dir, csv_path)
        print(f"[debug] Done. Inspect {crops_dir} and {csv_path}")
        return

    # -- normal evaluation
    results = {}
    engines = []
    if args.ocr in ('tesseract', 'both'):
        engines.append(('tesseract', ocr_tess))
    if args.ocr in ('easyocr', 'both'):
        engines.append(('easyocr', ocr_easy))

    for engine_name, ocr in engines:
        print(f"\n=== Evaluating OCR engine: {engine_name} ===")
        correct = 0
        times = []
        for fn, gt_plate in gt_entries:
            img_path = os.path.join(img_dir, fn)
            img = cv2.imread(img_path)
            start = time.time()
            crops = detector.detect_and_crop(img)
            # assume one plate per image; pick the highest‐score crop
            if crops:
                plate_img, _ = max(crops, key=lambda x: x[1])
                pred = ocr.recognize(plate_img)
            else:
                pred = ''
            elapsed = time.time() - start
            times.append(elapsed)
            ok = (pred == gt_plate)
            correct += ok
            print(f"{fn:8s} GT={gt_plate:10s} PRED={pred:10s} {'✓' if ok else '✗'}")
        total = len(gt_entries)
        acc  = correct / total * 100
        t100 = sum(times) / total * 100
        grade = calculate_final_grade(acc, t100)
        print(f"\n→ {engine_name}: Accuracy {acc:.2f}% ({correct}/{total}), "
              f"Time @100 imgs: {t100:.1f}s, Grade: {grade:.1f}\n")
        results[engine_name] = (acc, t100, grade)

    # final summary if single engine
    if args.ocr != 'both':
        acc, t100, grade = results[args.ocr]
        print(f"\nFinal ({args.ocr}): acc={acc:.2f}%, time100={t100:.1f}s, grade={grade:.1f}")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ocr',
        choices=['tesseract','easyocr','both'],
        default='both',
        help="Which OCR engine(s) to evaluate"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Run debug: save plate crops and batch-OCR them, then exit"
    )
    args = parser.parse_args()
    main(args)
