import itertools
import pandas as pd
from tqdm import tqdm

from detector import PlateDetector
from evaluation import evaluate
from config import CSV_PATH, IMAGES_DIR


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return [(r.fname, r.xtl, r.ytl, r.xbr, r.ybr, r.plate) for r in df.itertuples(index=False)]


def find_best_parameters(data, images_dir, num_threads: int = 4):
    """Brute-force search over a small grid of preprocessing parameters."""
    detector = PlateDetector()

    widths = [400, 450 ,500]
    bilateral_ds = [9, 11]
    block_sizes = [17, 19, 21, 23, 25]
    cs = [5, 10, 15, 20]
    confs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    crop_x1 = [0, 10, 15, 20, 25, 30]
    crop_x2 = [0, 10, 15, 20, 25, 30]
    crop_y1 = [0, 10, 15, 20, 25, 30]
    crop_y2 = [0, 10, 15, 20, 25, 30]

    thresh_methods = ["gaussian", "mean", "otsu"]
    deskew_apply = [True, False]
    psms = [6, 7, 8]
    oems = [3]

    total_combinations = (
            len(widths)
            * len(bilateral_ds)
            * len(block_sizes)
            * len(cs)
            * len(confs)
            * len(crop_x1)
            * len(crop_x2)
            * len(crop_y1)
            * len(crop_y2)
            * len(thresh_methods)
            * len(deskew_apply)
            * len(psms)
            * len(oems)
    )

    best_acc = -1.0
    best_params = {}

    param_product = itertools.product(
        widths,
        bilateral_ds,
        block_sizes,
        cs,
        confs,
        crop_x1,
        crop_x2,
        crop_y1,
        crop_y2,
        thresh_methods,
        deskew_apply,
        psms,
        oems,
    )

    for (w, d, bs, c_val, conf, x1, x2, y1, y2, t_m, d_apply, psm, oem) in tqdm(
            param_product, total=total_combinations, desc="Przeszukiwanie parametrów"
    ):
        preprocess = {
            "width": w,
            "bilateral_d": d,
            "block_size": bs,
            "c": c_val,
            "thresh_method": t_m,
            "deskew_apply": d_apply,
        }
        crop_off = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
        tess_config = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        acc, _, _ = evaluate(
            detector,
            data,
            images_dir,
            num_threads=num_threads,
            preprocess_params=preprocess,
            crop_offsets=crop_off,
            conf=conf,
            tesseract_config=tess_config,
        )
        if acc > best_acc:
            best_acc = acc
            best_params = {
                "preprocess": preprocess,
                "crop_offsets": crop_off,
                "conf": conf,
                "tesseract": tess_config,
            }
    return best_params, best_acc


if __name__ == "__main__":
    dataset = load_dataset(CSV_PATH)
    params, acc = find_best_parameters(dataset, IMAGES_DIR)
    print("Best accuracy", acc)
    print("Params", params)