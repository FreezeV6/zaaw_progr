import time
from utils.config import XML_PATH, IMAGES_DIR, CNN_MODEL_PATH, CROP_SIZE, TEST_RATIO, RANDOM_SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE
from utils.loader import load_cvat_xml
from detector.plate_detector import train_plate_detector, load_plate_detector, predict_box
from detector.ocr_reader import ocr_plate
from utils.image_utils import crop_plate_from_yolo, draw_box_on_img, convert_to_yolo_box, yolo_to_box
from utils.eval import calculate_accuracy, calculate_final_grade, calculate_iou
import cv2
import os

def main(train=True):
    train_set, test_set = load_cvat_xml(XML_PATH, IMAGES_DIR, TEST_RATIO, RANDOM_SEED)
    if train:
        print("== Trening modelu detektora ==")
        train_plate_detector(
            train_set, test_set,
            model_path=CNN_MODEL_PATH,
            crop_size=(224, 224),
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE
        )
    print("== Podgląd detekcji na pierwszych 10 zdjęciach ==")
    model = load_plate_detector(CNN_MODEL_PATH, crop_size=(224, 224))
    os.makedirs("debug_imgs", exist_ok=True)
    os.makedirs("debug_crops", exist_ok=True)
    for i, (img_path, true_box, true_plate, img_w, img_h) in enumerate(test_set[:10]):
        pred_box = predict_box(model, img_path, crop_size=(224, 224))
        plate_img = crop_plate_from_yolo(img_path, pred_box, img_w, img_h, out_size=CROP_SIZE)
        img = cv2.imread(img_path)
        img_pred = draw_box_on_img(img, pred_box, img_w, img_h, color=(0, 255, 0))
        img_all = draw_box_on_img(img_pred, convert_to_yolo_box(*true_box, img_w, img_h), img_w, img_h, color=(0, 0, 255))
        out_path = f"debug_imgs/debug_result_{i}.jpg"
        crop_out_path = f"debug_crops/ocr_crop_{i}_{true_plate}.jpg"
        cv2.imwrite(out_path, img_all)
        cv2.imwrite(crop_out_path, plate_img)
        pred_plate = ocr_plate(plate_img)
        print(f"GT: {true_plate} | OCR: {pred_plate}")
        print(f"Saved debug visualization: {out_path}")
    print("== Ewaluacja na zbiorze testowym ==")
    true_plates = []
    pred_plates = []
    ious = []
    start_time = time.time()
    for idx, (img_path, true_box, true_plate, img_w, img_h) in enumerate(test_set[:100]):
        pred_box = predict_box(model, img_path, crop_size=(224, 224))
        # do IoU: zamien pred_box na piksele!
        gt_box_pix = true_box
        pred_box_pix = yolo_to_box(*pred_box, img_w, img_h)
        iou = calculate_iou(gt_box_pix, pred_box_pix)
        ious.append(iou)
        plate_img = crop_plate_from_yolo(img_path, pred_box, img_w, img_h, out_size=CROP_SIZE)
        pred_plate = ocr_plate(plate_img)
        true_plates.append(true_plate)
        pred_plates.append(pred_plate)
    total_time = time.time() - start_time
    accuracy = calculate_accuracy(true_plates, pred_plates)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f'Dokładność OCR: {accuracy:.2f}%')
    print(f'IoU (średnie): {mean_iou:.3f}')
    print(f'Czas przetwarzania 100 zdjęć: {total_time:.2f}s')
    print(f'Ocena końcowa: {calculate_final_grade(accuracy, total_time)}')

if __name__ == '__main__':
    main(train=False)  # train=False jeśli nie chcesz ponownie trenować
