import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

from detector import PlateDetector
from evaluation import evaluate
from config import CSV_PATH, IMAGES_DIR


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return [(r.fname, r.xtl, r.ytl, r.xbr, r.ybr, r.plate) for r in df.itertuples(index=False)]


def find_best_parameters(data, images_dir, num_threads: int = 8, n_calls: int = 300, random_state: int = 42, n_random_starts: int = 20):
    detector = PlateDetector()

    # definiujemy przestrzeń przeszukiwania
    dims = [
        Categorical([500],      name="width"),
        Categorical([9, 11],     name="bilateral_d"),
        Categorical([17,19,21,23,25], name="block_size"),
        Categorical([5,10,15,20],    name="c"),
        Real(0.10, 0.35,         name="conf"),
        Integer(0, 40,           name="x1"),
        Integer(0, 40,           name="x2"),
        Integer(0, 40,           name="y1"),
        Integer(0, 40,           name="y2"),
        Categorical(["gaussian","mean","otsu"], name="thresh_method"),
        Categorical([False],            name="deskew_apply"),
        Categorical([6, 7, 8],                name="psm"),
        Categorical([3],                      name="oem"),
    ]

    @use_named_args(dims)
    def objective(**params):
        preprocess = {
            "width":         params["width"],
            "bilateral_d":   params["bilateral_d"],
            "block_size":    params["block_size"],
            "c":             params["c"],
            "thresh_method": params["thresh_method"],
            "deskew_apply":  params["deskew_apply"],
        }
        crop_off = {
            "x1": params["x1"],
            "x2": params["x2"],
            "y1": params["y1"],
            "y2": params["y2"],
        }
        tess_cfg = (
            f"--oem {params['oem']} --psm {params['psm']} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        acc, _, _ = evaluate(
            detector,
            data,
            images_dir,
            num_threads=num_threads,
            preprocess_params=preprocess,
            crop_offsets=crop_off,
            conf=params["conf"],
            tesseract_config=tess_cfg,
        )
        # gp_minimize minimalizuje, więc zwracamy -accuracy
        return -acc

    result = gp_minimize(
        func=objective,
        dimensions=dims,
        acq_func="EI",
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=random_state,
    )

    # odpakowujemy najlepsze wartości
    best_vals = dict(zip((d.name for d in dims), result.x))
    best_acc  = -result.fun

    best_params = {
        "preprocess": {
            "width":         best_vals["width"],
            "bilateral_d":   best_vals["bilateral_d"],
            "block_size":    best_vals["block_size"],
            "c":             best_vals["c"],
            "thresh_method": best_vals["thresh_method"],
            "deskew_apply":  best_vals["deskew_apply"],
        },
        "crop_offsets": {
            "x1": best_vals["x1"],
            "x2": best_vals["x2"],
            "y1": best_vals["y1"],
            "y2": best_vals["y2"],
        },
        "conf":     best_vals["conf"],
        "tesseract": (
            f"--oem {best_vals['oem']} --psm {best_vals['psm']} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    }

    return best_params, best_acc


if __name__ == "__main__":
    dataset = load_dataset(CSV_PATH)
    params, acc = find_best_parameters(
        dataset,
        IMAGES_DIR,
        num_threads=8,
        n_calls=300,  # 200 punktów zamiast 50
        n_random_starts=25,  # 25 całkowicie losowych startów
        random_state=42
    )
    print("Best accuracy:", acc)
    print("Params:", params)
