from ultralytics import YOLO
from config import MODEL_PATH


class PlateDetector:
    def __init__(self, model_path=MODEL_PATH, default_conf: float = 0.25, default_iou: float = 0.45):
        self.model = YOLO(model_path)
        self.default_conf = default_conf
        self.default_iou = default_iou

    def detect(self, image, conf: float | None = None, iou: float | None = None):
        conf_val = conf if conf is not None else self.default_conf
        iou_val = iou if iou is not None else self.default_iou
        results = self.model(image, conf=conf_val, iou=iou_val)
        if not results:
            return []
        # xyxy: (N,4), confidence, class
        preds = results[0].boxes.xyxy.cpu().numpy()
        return preds