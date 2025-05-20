from detector.ocr_reader import deskew
import cv2

def crop_plate_from_yolo(img_path, box, img_w, img_h, out_size=(224, 64)):
    x, y, w, h = box
    cx, cy = int(x * img_w), int(y * img_h)
    bw, bh = int(w * img_w), int(h * img_h)
    x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
    x2, y2 = min(img_w, cx + bw // 2), min(img_h, cy + bh // 2)
    img = cv2.imread(img_path)
    crop = img[y1:y2, x1:x2]
    # Usuwamy niebieski pasek "PL"
    h_crop, w_crop = crop.shape[:2]
    margin_x = int(0.11 * w_crop)
    margin_y = int(0.17 * h_crop)
    crop = crop[margin_y:h_crop-margin_y, margin_x:w_crop-margin_x]
    crop = cv2.resize(crop, out_size)
    crop = deskew(crop)
    return crop

def draw_box_on_img(img, box, img_w, img_h, color=(0,255,0)):
    x, y, w, h = box
    cx, cy = int(x * img_w), int(y * img_h)
    bw, bh = int(w * img_w), int(h * img_h)
    x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
    x2, y2 = min(img_w, cx + bw // 2), min(img_h, cy + bh // 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img
