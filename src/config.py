import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'photos')
TEST_DIR = os.path.join(DATA_DIR, 'test/images')
YOLO_LABELS_DIR = os.path.join(DATA_DIR, 'labels')
CSV_PATH = os.path.join(DATA_DIR, 'plates.csv')
ANNOT_PATH = os.path.join(DATA_DIR, 'annotations.xml')

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")  # zmień wg systemu
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train2", "weights", "best.pt")  # YOLO po treningu

TRUSTED_PLATES = ["XYZ1234", "ABC5678"]  # Przykład
SERVO_PIN = 17  # GPIO pin Raspberry Pi
CAMERA_INDEX = 0

PREPROCESS_PARAMS = {
    "width": 500,
    "bilateral_d": 11,
    "block_size": 25,
    "c": 10,
    "thresh_method": "otsu",
    "inv": True,
    "deskew_apply": False,
}

# Przycinanie wykrytej tablicy przed OCR
CROP_OFFSETS = {"x1": 35, "x2": 12, "y1": 1, "y2": 0}

# Minimalne prawdopodobieństwo wykrycia tablicy
CONFIDENCE_THRESHOLD = 0.23377228126973296

# Konfiguracja tesseracta używana w OCR
TESSERACT_CONFIG = (
    "--oem 3 --psm 8 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# ===== Ustawienia optymalizatora =====
OPTIMIZER_SETTINGS = {
    "n_calls": 300,
    "n_random_starts": 25,
    "random_state": 42,
}

OPTIMIZER_SPACE = {
    "width": (400,600),
    "bilateral_d": [9, 11],
    "block_size": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51],
    "c": (0, 50),
    "conf": (0, 0.7),
    "x1": (0, 40),
    "x2": (0, 40),
    "y1": (0, 40),
    "y2": (0, 40),
    "thresh_method": ["gaussian", "mean", "otsu"],
    "inv": [False, True],
    "deskew_apply": [False],
    "psm": [8],
    "oem": [3],
}