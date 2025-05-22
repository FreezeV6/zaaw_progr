import cv2
import numpy as np
import onnxruntime as ort

class PlateDetector:
    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 input_size: int = 640):
        """
        model_path: ścieżka do pliku .onnx
        conf_threshold: minimalne zaufanie do detekcji
        iou_threshold: próg NMS
        input_size: rozmiar kwadratowego wejścia
        """
        self.session = ort.InferenceSession(model_path,
                                            providers=['CPUExecutionProvider'])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # [1,3,H,W]
        _, _, self.h, self.w = inp.shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

    def _letterbox(self, image):
        h0, w0 = image.shape[:2]
        r = self.input_size / max(h0, w0)
        new_w, new_h = int(w0 * r), int(h0 * r)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = self.input_size - new_w, self.input_size - new_h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114,114,114))
        # BGR→RGB, float32 0–1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # HWC→NCHW
        img = np.transpose(img, (2,0,1))[None, ...]
        return img, r, left, top

    def _nms(self, boxes, scores):
        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold
        )
        if len(idxs) == 0:
            return []
        # cv2.dnn returns list of [[i],...]
        return [i[0] if isinstance(i, (list,tuple,np.ndarray)) else i for i in idxs]

    def detect(self, image: np.ndarray):
        """
        image: BGR numpy array
        zwraca listę detekcji: [ ((x1,y1,x2,y2), score, class_id), ... ]
        """
        h0, w0 = image.shape[:2]
        img, r, pad_x, pad_y = self._letterbox(image)
        outputs = self.session.run(None, {self.input_name: img})
        pred = outputs[0][0]  # shape (N,6): x1,y1,x2,y2,score,class
        # filtrowanie po pewności
        mask = pred[:, 4] > self.conf_threshold
        pred = pred[mask]

        if pred.shape[0] == 0:
            return []

        # odczyt surowych boxów i score/class
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classes = pred[:, 5].astype(int)

        # przeskalowanie do oryginalnego obrazu
        # odjąć padding, podzielić przez r
        boxes[:, [0,2]] = (boxes[:, [0,2]] - pad_x) / r
        boxes[:, [1,3]] = (boxes[:, [1,3]] - pad_y) / r

        # NMS
        keep = self._nms(boxes, scores)
        detections = []
        for i in keep:
            x1,y1,x2,y2 = boxes[i]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)
            detections.append((
                (int(x1), int(y1), int(x2), int(y2)),
                float(scores[i]),
                int(classes[i])
            ))
        return detections
