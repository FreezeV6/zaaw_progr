import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'photos')
TEST_DIR = os.path.join(DATA_DIR, 'test/images')
YOLO_LABELS_DIR = os.path.join(DATA_DIR, 'labels')
CSV_PATH = os.path.join(DATA_DIR, 'plates.csv')
ANNOT_PATH = os.path.join(DATA_DIR, 'annotations.xml')

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # zmień wg systemu
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train2", "weights", "best.pt")  # YOLO po treningu

TRUSTED_PLATES = ["XYZ1234", "ABC5678"]  # Przykład
SERVO_PIN = 17  # GPIO pin Raspberry Pi
CAMERA_INDEX = 0
