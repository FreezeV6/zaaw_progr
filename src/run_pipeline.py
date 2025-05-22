import os, time, re
from src import prepare_data, detector, ocr_engine

# 1. Prepare dataset (download and split) if not already done
if not os.path.isdir('data/images/test'):
    prepare_data.prepare_dataset()

# 2. Load ground truth for test images
gt_file = 'data/test_ground_truth.txt'
if not os.path.exists(gt_file):
    raise FileNotFoundError("Ground truth file not found. Please run prepare_data.py first.")
test_entries = []
with open(gt_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        file_name = parts[0]
        plate_text = parts[1]
        # Parse ground truth bounding box coordinates
        xtl, ytl, xbr, ybr = map(float, parts[2:])
        test_entries.append({
            "file": file_name,
            "text": plate_text,
            "bbox": (xtl, ytl, xbr, ybr)
        })

# 3. Initialize the YOLO plate detector (uses CPU or GPU depending on availability)
plate_detector = detector.PlateDetector(weight_path='model/best.pt')

# Counters for evaluation
total_images = len(test_entries)
easy_correct = 0
tess_correct = 0
detections_count = 0
iou_sum = 0.0

# 4. Run detection + EasyOCR on each test image, measure accuracy and IoU
start_time = time.time()
for entry in test_entries:
    img_path = os.path.join('data/images/test', entry["file"])
    plate_img, pred_bbox = plate_detector.detect_plate(img_path)
    # Perform OCR with EasyOCR
    pred_text_easy = "" if plate_img is None else ocr_engine.ocr_easy(plate_img)
    # Normalize predictions and ground truth for comparison (remove spaces, uppercase)
    pred_text_easy_norm = re.sub(r'\s+', '', pred_text_easy).upper()
    true_text_norm = entry["text"].strip().upper()
    if pred_text_easy_norm == true_text_norm:
        easy_correct += 1
    # If a plate was detected, evaluate detection quality (IoU)
    if pred_bbox is not None:
        detections_count += 1
        # Calculate IoU between predicted bbox and ground truth bbox
        x1_p, y1_p, x2_p, y2_p = pred_bbox
        x1_g, y1_g, x2_g, y2_g = entry["bbox"]
        inter_x1 = max(x1_p, x1_g); inter_y1 = max(y1_p, y1_g)
        inter_x2 = min(x2_p, x2_g); inter_y2 = min(y2_p, y2_g)
        if inter_x2 >= inter_x1 and inter_y2 >= inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0.0
        pred_area = (x2_p - x1_p) * (y2_p - y1_p)
        gt_area = (x2_g - x1_g) * (y2_g - y1_g)
        union_area = pred_area + gt_area - inter_area
        iou_val = inter_area / union_area if union_area > 0 else 0.0
        iou_sum += iou_val
end_time = time.time()
easy_time = end_time - start_time  # total time for EasyOCR pipeline

# 5. Run detection + Tesseract OCR on each test image
start_time = time.time()
for entry in test_entries:
    img_path = os.path.join('data/images/test', entry["file"])
    plate_img, pred_bbox = plate_detector.detect_plate(img_path)
    pred_text_tess = "" if plate_img is None else ocr_engine.ocr_tesseract(plate_img)
    pred_text_tess_norm = re.sub(r'\s+', '', pred_text_tess).upper()
    true_text_norm = entry["text"].strip().upper()
    if pred_text_tess_norm == true_text_norm:
        tess_correct += 1
    # (We assume detection outcomes are the same as above, so IoU/detection count not recalculated here)
end_time = time.time()
tess_time = end_time - start_time  # total time for Tesseract pipeline

# 6. Calculate accuracy metrics
easy_accuracy = (easy_correct / total_images) * 100.0  # percentage of plates read correctly by EasyOCR
tess_accuracy = (tess_correct / total_images) * 100.0  # percentage of plates read correctly by Tesseract

# Calculate detection performance metrics
mean_iou = (iou_sum / detections_count) if detections_count > 0 else 0.0
detection_rate = (detections_count / total_images) * 100.0  # percentage of images where a plate was detected

# 7. Define the grading function as per project specification
def calculate_final_grade(accuracy_percent, processing_time_sec):
    """
    Calculates the final grade based on OCR accuracy and processing time for 100 images.
    Returns a grade on a 2.0 - 5.0 scale (rounded to nearest 0.5).
    """
    # Minimum requirements: accuracy >= 60% and time <= 60s for 100 images
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    # Normalize metrics to [0, 1] range
    acc_norm = (accuracy_percent - 60) / 40.0  # 60% -> 0.0, 100% -> 1.0
    time_norm = (60.0 - processing_time_sec) / 60.0  # 60s -> 0.0, 0s -> 1.0
    if time_norm < 0:
        time_norm = 0.0
    # Weighted sum (0.7 accuracy, 0.3 time) mapped to [2.0, 5.0]
    score = 2.0 + 3.0 * (0.7 * acc_norm + 0.3 * time_norm)
    return round(score * 2) / 2  # round to nearest 0.5

# Extrapolate processing time to 100 images (if test set is not 100 images)
easy_time_100 = easy_time * (100.0 / total_images)
tess_time_100 = tess_time * (100.0 / total_images)
easy_grade = calculate_final_grade(easy_accuracy, easy_time_100)
tess_grade = calculate_final_grade(tess_accuracy, tess_time_100)

# 8. Print the evaluation report
print(f"Total test images: {total_images}")
print(f"Detection success rate: {detection_rate:.1f}%  (detected plates in {detections_count}/{total_images} images)")
print(f"Mean IoU of detected plates: {mean_iou:.3f}")
print(f"EasyOCR OCR Accuracy: {easy_accuracy:.2f}%  ({easy_correct}/{total_images} correct)")
print(f"Tesseract OCR Accuracy: {tess_accuracy:.2f}%  ({tess_correct}/{total_images} correct)")
print(f"EasyOCR total processing time for {total_images} images: {easy_time:.2f} seconds")
print(f"Tesseract total processing time for {total_images} images: {tess_time:.2f} seconds")
print(f"Average time per image: EasyOCR = {easy_time/total_images:.3f}s, Tesseract = {tess_time/total_images:.3f}s")
print(f"Final grade (EasyOCR pipeline): {easy_grade:.1f} / 5.0")
print(f"Final grade (Tesseract pipeline): {tess_grade:.1f} / 5.0")
