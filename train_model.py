from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='utils/data.yaml',
    epochs=20,
    imgsz=640,
    cache = True,
    workers = 8
)