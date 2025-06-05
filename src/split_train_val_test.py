import os
import shutil
import random
from config import IMAGES_DIR, YOLO_LABELS_DIR

random.seed(42)

def split_files(img_dir, label_dir, out_base, train=0.60, val=0.2):
    files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train)
    n_val = int(n * val)
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }
    for split, flist in splits.items():
        img_out = os.path.join(out_base, split, "images")
        lbl_out = os.path.join(out_base, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for f in flist:
            shutil.copy(os.path.join(img_dir, f), os.path.join(img_out, f))
            shutil.copy(os.path.join(label_dir, f.replace(".jpg", ".txt")), os.path.join(lbl_out, f.replace(".jpg", ".txt")))
    print("Split done:", {k: len(v) for k, v in splits.items()})

split_files(IMAGES_DIR, YOLO_LABELS_DIR, "../data/")
