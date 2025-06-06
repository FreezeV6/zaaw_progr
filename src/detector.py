from ultralytics import YOLO
from config import MODEL_PATH


class PlateDetector:
    def __init__(self, model_path=MODEL_PATH, default_conf: float = 0.25):
        self.model = YOLO(model_path)
        self.default_conf = default_conf

    def detect(self, image, conf: float | None = None):
        conf_val = conf if conf is not None else self.default_conf
        results = self.model(image, conf=conf_val)
        if not results:
            return []
        # xyxy: (N,4), confidence, class
        preds = results[0].boxes.xyxy.cpu().numpy()
        return preds