try:
    # lekki interpreter na Raspberry Pi
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # fallback na pełne TensorFlow na PC/Windows
    from tensorflow.lite import Interpreter

import numpy as np
import cv2

class PlateDetector:
    def __init__(self, model_path: str, threshold: float = 0.25):
        # Interpreter to teraz klasa z jednego z dwóch modułów
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.threshold = threshold
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image: np.ndarray):
        img = cv2.resize(image, tuple(self.input_details[0]['shape'][2:0:-1]))
        inp = img.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        h, w = image.shape[:2]
        boxes = []
        for x1,y1,x2,y2,conf,cls in preds:
            if conf < self.threshold:
                continue
            boxes.append([x1 * w, y1 * h, x2 * w, y2 * h, conf])
        return boxes
