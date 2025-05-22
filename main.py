import time
import logging
from modules import dataset, detector, ocr, utils

# Konfiguracja logowania
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

def main():
    # Ścieżki do danych wejściowych
    photos_dir = "data/photos"
    annotations_file = "data/annotations.xml"
    # 1. Przygotowanie danych
    logging.info("Loading annotations from XML...")
    data_list = dataset.load_annotations(annotations_file)
    logging.info(f"Total images with plates: {len(data_list)}")
    train_data, test_data = dataset.split_data(data_list, train_ratio=0.7)
    logging.info(f"Split into {len(train_data)} training and {len(test_data)} testing examples.")
    data_yaml = dataset.prepare_yolo_dataset(train_data, test_data, base_dir="data")
    # 2. Trenowanie modelu detekcji tablic
    model = detector.train_detector()
    # 3. Detekcja na zbiorze testowym
    test_image_paths = [f"data/test/images/{item['filename']}" for item in test_data]
    logging.info(f"Running detection on {len(test_image_paths)} test images...")
    start_time = time.time()
    detections = detector.detect_plates(model, test_image_paths, imgsz=640, iou=0.5, conf=0.5)
    detection_time = time.time() - start_time
    logging.info(f"Detection on test set completed in {detection_time:.2f} seconds.")
    # 4. OCR na wykrytych tablicach i obliczanie metryk
    correct_count = 0
    total_count = 0
    iou_values = []
    # Przygotuj plik CSV na wyniki
    import csv
    results_file = "results.csv"
    with open(results_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "ground_truth", "predicted_text", "IoU", "correct"])
        # Iteruj po wszystkich obrazach testowych i odpowiadających detekcjach
        for item, det in zip(test_data, detections):
            filename = item['filename']
            true_text = item['plate_text'] or ""
            true_text = true_text.strip().upper()
            true_bbox = tuple(map(int, item['bbox']))
            pred_text = ""
            pred_bbox = None
            if det['boxes']:
                # Jeśli wykryto conajmniej jedną tablicę, weź tę z najwyższym score (pierwsza na liście)
                pred_bbox = det['boxes'][0]
                pred_text = ocr.recognize_plate_text(f"dataset/test/images/{filename}", pred_bbox)
            else:
                # Jeśli brak detekcji - pozostaw pred_text pusty, pred_bbox = None
                pred_bbox = None
            # Oblicz IoU między pred_bbox a true_bbox (jeśli detection istnieje)
            iou = 0.0
            if pred_bbox is not None:
                iou = utils.calculate_iou(pred_bbox, true_bbox)
            iou_values.append(iou)
            # Sprawdź poprawność OCR (pełny tekst musi się zgadzać)
            is_correct = (pred_text == true_text and pred_text != "")
            if is_correct:
                correct_count += 1
            total_count += 1
            # Zapisz do pliku CSV wynik dla tej próbki
            writer.writerow([filename, true_text, pred_text, f"{iou:.3f}", int(is_correct)])
            logging.debug(f"Processed {filename}: GT='{true_text}', Pred='{pred_text}', IoU={iou:.2f}, Correct={is_correct}")
    logging.info(f"Saved detailed results to {results_file}")
    # 5. Obliczanie zbiorczych metryk
    accuracy = (correct_count / total_count) * 100.0 if total_count > 0 else 0.0
    avg_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0
    # Przeskaluj zmierzony czas detekcji+OCR do 100 obrazów
    num_images = len(test_data)
    time_per_image = detection_time / num_images if num_images > 0 else 0
    time_100 = time_per_image * 100.0
    # Wylicz ocenę końcową za pomocą zdefiniowanej funkcji
    final_grade = utils.calculate_final_grade(accuracy, time_100)
    # 6. Wyświetlenie podsumowania
    logging.info(f"OCR Accuracy: {accuracy:.2f}%")
    logging.info(f"Average IoU (detection): {avg_iou:.3f}")
    logging.info(f"Total processing time for {num_images} test images: {detection_time:.2f} s")
    logging.info(f"Estimated time for 100 images: {time_100:.2f} s")
    logging.info(f"Final Grade (scale 2.0-5.0): {final_grade:.1f}")
    print("========== ALPR System Evaluation ==========")
    print(f"Test images: {num_images}")
    print(f"OCR Accuracy: {accuracy:.2f}%")
    print(f"Average IoU: {avg_iou:.3f}")
    print(f"Processing time for {num_images} images: {detection_time:.2f} s")
    print(f"Estimated time for 100 images: {time_100:.2f} s")
    print(f"Final Grade: {final_grade:.1f}")
    print("=============================================")

if __name__ == "__main__":
    main()
