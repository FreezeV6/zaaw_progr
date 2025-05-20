import cv2

def crop_plate_from_yolo(img_path, box, img_w, img_h, out_size=(128, 64)):
    img = cv2.imread(img_path)
    x_center, y_center, bw, bh = box
    x1 = int((x_center - bw / 2) * img_w)
    y1 = int((y_center - bh / 2) * img_h)
    x2 = int((x_center + bw / 2) * img_w)
    y2 = int((y_center + bh / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(int(img_w), x2), min(int(img_h), y2)
    plate_img = img[y1:y2, x1:x2]
    # Resize
    plate_img = cv2.resize(plate_img, out_size)
    return plate_img

def draw_box_on_img(img, box, img_w, img_h, color=(0,255,0)):
    x_center, y_center, bw, bh = box
    x1 = int((x_center - bw / 2) * img_w)
    y1 = int((y_center - bh / 2) * img_h)
    x2 = int((x_center + bw / 2) * img_w)
    y2 = int((y_center + bh / 2) * img_h)
    img_copy = img.copy()
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
    return img_copy
