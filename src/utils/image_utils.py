import cv2
import numpy as np

def convert_to_yolo_box(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return [x_center, y_center, width, height]

def yolo_to_box(x_center, y_center, width, height, img_w, img_h):
    xmin = int((x_center - width/2) * img_w)
    xmax = int((x_center + width/2) * img_w)
    ymin = int((y_center - height/2) * img_h)
    ymax = int((y_center + height/2) * img_h)
    # safety
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w-1, xmax)
    ymax = min(img_h-1, ymax)
    return [xmin, ymin, xmax, ymax]

def crop_plate_from_yolo(img_path, box, img_w, img_h, out_size=(224, 64)):
    # box: [x_center, y_center, width, height] w [0,1]
    xmin, ymin, xmax, ymax = yolo_to_box(*box, img_w, img_h)
    img = cv2.imread(img_path)
    crop = img[ymin:ymax, xmin:xmax]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return np.zeros(out_size, dtype=np.uint8)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, out_size)
    # Pre-processing dla OCR:
    crop = cv2.equalizeHist(crop)
    _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return crop

def draw_box_on_img(img, box, img_w, img_h, color=(0, 255, 0)):
    # box: [x_center, y_center, width, height] w [0,1]
    xmin, ymin, xmax, ymax = yolo_to_box(*box, img_w, img_h)
    img_box = img.copy()
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    return img_box
