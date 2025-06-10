import os
import cv2
import random

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

CHARS_MAP = {"1": "I", "2": "Z", "3": "C", "5": "S", "0": "O", "7": "Z"}
REV_CHARS_MAP = {"I": "1", "Z": "2", "C": "3", "S": "5", "O": "0"}

PREPROCESS_PARAMS = {
    "width": 768,
    "blur_method": "gaussian",
    "gaussian_kernel": (3, 3),
    "bilateral_d": 5,
    "sigma_color": 16.438717857645447,
    "sigma_space": 75.0,
    "gamma": 2.5,
    "clahe_clip": 1.039094473075576,
    "clahe_tile_grid": (4, 4),
    "block_size": 29,
    "adaptive_block": 29,
    "adaptive_C": 0,
    "c": 15,
    "thresh_method": "otsu",
    "inv": False,
    "deskew_apply": False,
    "deskew_border": 1,
    "kernel_size": 9,
    "open_iter":1,
    "close_iter": 0,
    "dilate_iter": 0,
}

LIVE_PREPROCESS_PARAMS = {
    "width": 768,
    "blur_method": "gaussian",
    "gaussian_kernel": (3, 3),
    "bilateral_d": 5,
    "sigma_color": 16.438717857645447,
    "sigma_space": 75.0,
    "gamma": 1.2,
    "clahe_clip": 1.039094473075576,
    "clahe_tile_grid": (4, 4),
    "block_size": 29,
    "adaptive_block": 29,
    "adaptive_C": 0,
    "c": 15,
    "thresh_method": "otsu",
    "inv": False,
    "deskew_apply": False,
    "deskew_border": 1,
    "kernel_size": 9,
    "open_iter":1,
    "close_iter": 0,
    "dilate_iter": 0,
}

# Przycinanie wykrytej tablicy przed OCR
CROP_OFFSETS = {"x1": 34, "x2": 10, "y1": 0, "y2": 0}
SHRINK_RATIO = 0.0

LIVE_CROP_OFFSETS = {"x1": -10, "x2": -10, "y1": -10, "y2": -10}
LIVE_SHRINK_RATIO = 0.0

# Minimalne prawdopodobieństwo wykrycia tablicy
YOLO_CONFIDENCE = 0.4481218981217491
LIVE_YOLO_CONFIDENCE = 0.25
YOLO_NMS_IOU = 0.4047360392815532
SEED = [3, 4, 13, 15, 16, 21, 28, 36, 38, 39]
OCR_CONF_MIN = 0.1921366962440251

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
    "deskew_apply": [False],
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
    "inv": [False],

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

LIVE_OPTIMIZER_SPACE = {
    # --- DETECTION ---
    "yolo_conf": (0.1, 0.5),
    "yolo_nms": (0.3, 0.6),

    # --- CROPPING/SHRINK ---
    "shrink_ratio": (0.0, 0.2),

    # --- DESKEW ---
    "deskew_apply": [False],
    "deskew_border": [cv2.BORDER_CONSTANT],

    # --- RESIZE ---
    "width": (300, 800),

    # --- CLAHE ---
    "clahe_clip": (1.0, 10.0),
    "clahe_tile_grid": [(8, 8), (16, 16)],

    # --- GAMMA ---
    "gamma": (0.5, 2.5),

    # --- BLUR ---
    "blur_method": ["gaussian", "bilateral"],
    "gaussian_kernel": [(3, 3), (5, 5)],
    "bilateral_d": [5, 9],
    "sigma_color": (15, 75),
    "sigma_space": (15, 75),

    # --- THRESHOLDING ---
    "thresh_method": ["otsu", "adaptive"],
    "adaptive_block": [3,15, 31],
    "adaptive_C": (-10, 10),
    "inv": [False],

    # --- MORPHOLOGY ---
    "kernel_size": [3, 5],
    "open_iter": [0],
    "close_iter": [0],
    "dilate_iter": [0],

    # --- TESSERACT CONFIG ---
    "psm": [8],
    "oem": [3],

    # --- OCR CONFIDENCE FILTER ---
    "ocr_conf_min": (0.0, 0.7),
}

rand = random.Random(random.choice(SEED))