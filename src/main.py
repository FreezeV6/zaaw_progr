import os
import yaml
from tqdm import tqdm
from src.detector import PlateDetector
from src.ocr_reader import PlateOCR
from src.utils.eval import calculate_iou, fuzzy_match
from src.utils.logger import setup_logger
from src.data_loader import load_cvat_xml

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    logger = setup_logger()
    logger.info("Start programu...")

    # Wczytaj dane (ścieżka do XML i folderu zdjęć!)
    all_data = load_cvat_xml(config['annotations_xml'], config['photos_dir'])

    detector = PlateDetector(config['yolo_model'])
    ocr = PlateOCR(mode=config['ocr_mode'], psm=config['ocr_psm'], whitelist=config['ocr_whitelist'])
    os.makedirs(config['output_dir'], exist_ok=True)
    results = []
    correct_ocr = 0
    ious = []
    for img_path, gt_box, gt_plate in tqdm(all_data, desc="ANPR"):
        import cv2
        img = cv2.imread(img_path)
        box = detector.detect(img)
        if box is None:
            logger.warning(f"No detection for {img_path}")
            continue
        crop = detector.crop_plate(img, box)
        pred_text = ocr.recognize(crop)
        iou = calculate_iou(gt_box, box)
        score = fuzzy_match(pred_text, gt_plate)
        results.append((os.path.basename(img_path), pred_text, gt_plate, iou, score))
        ious.append(iou)
        if score > config['fuzzy_match_threshold']:
            correct_ocr += 1
        logger.info(f"[{img_path}] Pred: {pred_text}, GT: {gt_plate}, IOU: {iou:.3f}, OCR Score: {score:.2f}")
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    ocr_acc = correct_ocr / len(all_data)
    logger.info(f"OCR Accuracy: {ocr_acc*100:.1f}% | Mean IoU: {mean_iou:.3f}")

    # Zapisz wyniki
    with open(os.path.join(config['output_dir'], "results.txt"), "w") as f:
        for row in results:
            f.write(f"{row}\n")

if __name__ == "__main__":
    main()
