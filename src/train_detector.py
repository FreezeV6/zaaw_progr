import argparse
from ultralytics import YOLO

def train(data_yaml: str, epochs: int, imgsz: int, batch: int, model_size: str):
    """
    Trenuje YOLO (np. yolov8n.pt) na Twoich danych.
    """
    model = YOLO(f'yolov8{model_size}.pt')  # 'n' = nano, 's' = small
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='plate-detector'
    )
    # Eksport do TFLite
    model.export(format='tflite')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    type=str, default='data/data.yaml')
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--imgsz',   type=int, default=640)
    parser.add_argument('--batch',   type=int, default=8)
    parser.add_argument('--size',    type=str, default='n', choices=['n','s'])
    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.size)
