import cv2
import numpy as np

def yolo2abs(box, img_w, img_h):
    x_c, y_c, w, h = box
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h
    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)
    return x1, y1, x2, y2

def crop_plate_from_yolo(img_path, yolo_box, img_w, img_h, out_size=(224, 64), scale=1.2):
    img = cv2.imread(img_path)
    cx, cy, w, h = yolo_box
    # Skalowanie boxa
    w = w * scale
    h = h * scale
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    # Upewnij się, że box nie wychodzi poza obraz
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, out_size)
    return crop


def draw_box_on_img(img, box, img_w, img_h, color=(0,255,0)):
    x1, y1, x2, y2 = yolo2abs(box, img_w, img_h)
    img = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), color, 2)
    return img
