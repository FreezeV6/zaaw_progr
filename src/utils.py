import cv2

def crop_image(image, bbox):
    """
    Wycinanie roi z obrazu: bbox = [x1,y1,x2,y2,...]
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    return image[y1:y2, x1:x2]

def compute_iou(boxA, boxB):
    """
    Oblicza IoU dwóch boxów [x1,y1,x2,y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
