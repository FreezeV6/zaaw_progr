import os
import cv2

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
    "blur_method": "bilateral",
    "gaussian_kernel": (5, 5),
    "bilateral_d": 9,
    "sigma_color": 41,
    "sigma_space": 21,
    "gamma": 1.0,
    "clahe_clip": 2.0,
    "clahe_tile_grid": (8, 8),
    "block_size": 25,
    "adaptive_block": 25,
    "adaptive_C": 0,
    "c": 15,
    "thresh_method": "gaussian",
    "inv": False,
    "deskew_apply": True,
    "deskew_border": cv2.BORDER_REPLICATE,
    "kernel_size": 3,
    "open_iter": 0,
    "close_iter": 0,
    "dilate_iter": 0,
}

# Przycinanie wykrytej tablicy przed OCR
CROP_OFFSETS = {"x1": 34, "x2": 10, "y1": 0, "y2": 0}
SHRINK_RATIO = 0.05

# Minimalne prawdopodobieństwo wykrycia tablicy
YOLO_CONFIDENCE = 0.4
YOLO_NMS_IOU = 0.45

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
    # --- DETECTION ---
    "yolo_conf": (0.1, 0.5),
    "yolo_nms": (0.3, 0.6),

    # --- CROPPING/SHRINK ---
    "shrink_ratio": (0.0, 0.2),

    # --- DESKEW ---
    "deskew_apply": [False, True],
    "deskew_border": [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE],

    # --- RESIZE ---
    "width": (300, 800),

    # --- CLAHE ---
    "clahe_clip": (1.0, 10.0),
    "clahe_tile_grid": [(4, 4), (8, 8), (16, 16), (32, 32)],

    # --- GAMMA ---
    "gamma": (0.5, 2.5),

    # --- BLUR ---
    "blur_method": ["gaussian", "bilateral"],
    "gaussian_kernel": [(3, 3), (5, 5), (7, 7)],
    "bilateral_d": [5, 7, 9, 11, 13],
    "sigma_color": (15, 75),
    "sigma_space": (15, 75),

    # --- THRESHOLDING ---
    "thresh_method": ["otsu", "gaussian", "mean", "adaptive"],
    "adaptive_block": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51],
    "adaptive_C": (-10, 10),
    "inv": [False, True],

    # --- MORPHOLOGY ---
    "kernel_size": [3, 5, 7, 9, 11],
    "open_iter": [0, 1, 2, 3],
    "close_iter": [0, 1, 2, 3],
    "dilate_iter": [0, 1, 2, 3],

    # --- TESSERACT CONFIG ---
    "psm": [8],
    "oem": [3],

    # --- OCR CONFIDENCE FILTER ---
    "ocr_conf_min": (0.0, 0.7),
}