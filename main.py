import os
import time
import csv
import argparse
import logging

from modules import dataset, detector, ocr, utils

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

def parse_args():
    p = argparse.ArgumentParser(description="ALPR: train + eval or eval-only")
    p.add_argument("--eval-only", action="store_true",
                   help="Pomiń trening, od razu wczytaj wytrenowany model i wykonaj ewaluację")
    p.add_argument("--weights", type=str, default=None,
                   help="Ręczna ścieżka do pliku wag .pt (tylko przy --eval-only)")
    return p.parse_args()

def main():
    args = parse_args()

    # 1. Przygotowanie danych
    logging.info("Loading and preparing dataset...")
    data = dataset.prepare_dataset()
    train_data, test_data = data['train'], data['test']
    logging.info(f"Train images: {len(train_data)}, Test images: {len(test_data)}")

    # 2. Trenowanie lub wczytanie modelu
    if args.eval_only:
        logging.info("Eval-only: loading existing weights...")
        model = detector.load_detector(weights_path=args.weights)
    else:
        logging.info("Starting YOLOv8 training...")
        weights = detector.train_detector()
        logging.info(f"Training done, weights saved to {weights}")
        model = detector.load_detector(weights_path=weights)

    # 3. Detekcja + OCR na zbiorze testowym
    test_paths = [os.path.join("data", "test", "images", item['filename']) for item in test_data]
    logging.info(f"Running detection on {len(test_paths)} test images...")
    start = time.time()
    detections = detector.detect_plates(model, test_paths)
    detection_time = time.time() - start
    logging.info(f"Detection+OCR time: {detection_time:.2f}s for {len(test_paths)} images")

    # 4. OCR + metryki
    total, correct, sum_iou, matched = 0, 0, 0.0, 0
    results = []
    for item in test_data:
        fname = item['filename']
        gt_boxes = item['bboxes']
        gt_texts = item['texts']
        preds = detections.get(fname, [])
        used = set()
        for i, gt_box in enumerate(gt_boxes):
            total += 1
            best_iou, best_j = 0.0, None
            for j, pred in enumerate(preds):
                if j in used: continue
                iou = utils.calculate_iou(gt_box, pred['bbox'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None:
                used.add(best_j)
                matched += 1
                pred_box = preds[best_j]['bbox']
                text = ocr.recognize_plate(os.path.join("data","test","images", fname), pred_box)
                correct_flag = int(text.replace(" ", "").upper() == gt_texts[i].replace(" ", "").upper())
                correct += correct_flag
                sum_iou += best_iou
                results.append([fname, gt_texts[i], text, f"{best_iou:.2f}", str(correct_flag)])
            else:
                results.append([fname, gt_texts[i], "", "0.00", "0"])

    # 5. Podsumowanie i zapis CSV
    acc = utils.calculate_accuracy(correct, total)
    avg_iou = sum_iou / matched if matched else 0.0
    time100 = (detection_time / len(test_paths)) * 100.0 if test_paths else 0.0
    grade = utils.calculate_final_grade(acc, time100)

    # Zapis szczegółowego pliku
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filename","gt","pred","IoU","correct"])
        w.writerows(results)

    print("=== EVALUATION SUMMARY ===")
    print(f"Total plates: {total}")
    print(f"Correct OCR: {correct} ({acc:.2f}%)")
    print(f"Average IoU: {avg_iou:.2f}")
    print(f"Time for {len(test_paths)} images: {detection_time:.2f}s, est. {time100:.2f}s/100")
    print(f"Final grade: {grade:.1f}")

if __name__ == "__main__":
    main()
