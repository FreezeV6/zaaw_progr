import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

from detector import PlateDetector
from evaluation import evaluate
from config import (
    CSV_PATH,
    IMAGES_DIR,
    OPTIMIZER_SETTINGS,
    OPTIMIZER_SPACE,
    CROP_OFFSETS,
)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return [(r.fname, r.xtl, r.ytl, r.xbr, r.ybr, r.plate) for r in df.itertuples(index=False)]


def find_best_parameters(
    data,
    images_dir,
    num_threads: int = 8,
    n_calls: int | None = None,
    random_state: int | None = None,
    n_random_starts: int | None = None,
):
    """Przeszukaj przestrzeń parametrów i zwróć najlepszy zestaw."""

    detector = PlateDetector()

    # parametry optymalizacji z config, jeśli nie podano inaczej
    if n_calls is None:
        n_calls = OPTIMIZER_SETTINGS.get("n_calls", 50)
    if random_state is None:
        random_state = OPTIMIZER_SETTINGS.get("random_state", 42)
    if n_random_starts is None:
        n_random_starts = OPTIMIZER_SETTINGS.get("n_random_starts", 10)

    # definiujemy przestrzeń przeszukiwania na podstawie config
    dims = [
        Real(*OPTIMIZER_SPACE["yolo_conf"], name="yolo_conf"),
        Real(*OPTIMIZER_SPACE["yolo_nms"], name="yolo_nms"),
        Real(*OPTIMIZER_SPACE["shrink_ratio"], name="shrink_ratio"),
        Categorical(OPTIMIZER_SPACE["deskew_apply"], name="deskew_apply"),
        Categorical(OPTIMIZER_SPACE["deskew_border"], name="deskew_border"),
        Integer(*OPTIMIZER_SPACE["width"], name="width"),
        Real(*OPTIMIZER_SPACE["clahe_clip"], name="clahe_clip"),
        Categorical(OPTIMIZER_SPACE["clahe_tile_grid"], name="clahe_tile_grid"),
        Real(*OPTIMIZER_SPACE["gamma"], name="gamma"),
        Categorical(OPTIMIZER_SPACE["blur_method"], name="blur_method"),
        Categorical(OPTIMIZER_SPACE["gaussian_kernel"], name="gaussian_kernel"),
        Categorical(OPTIMIZER_SPACE["bilateral_d"], name="bilateral_d"),
        Real(*OPTIMIZER_SPACE["sigma_color"], name="sigma_color"),
        Real(*OPTIMIZER_SPACE["sigma_space"], name="sigma_space"),
        Categorical(OPTIMIZER_SPACE["thresh_method"], name="thresh_method"),
        Categorical(OPTIMIZER_SPACE["adaptive_block"], name="adaptive_block"),
        Integer(*OPTIMIZER_SPACE["adaptive_C"], name="adaptive_C"),
        Categorical(OPTIMIZER_SPACE["inv"], name="inv"),
        Categorical(OPTIMIZER_SPACE["kernel_size"], name="kernel_size"),
        Categorical(OPTIMIZER_SPACE["open_iter"], name="open_iter"),
        Categorical(OPTIMIZER_SPACE["close_iter"], name="close_iter"),
        Categorical(OPTIMIZER_SPACE["dilate_iter"], name="dilate_iter"),
        Categorical(OPTIMIZER_SPACE["psm"], name="psm"),
        Categorical(OPTIMIZER_SPACE["oem"], name="oem"),
        Real(*OPTIMIZER_SPACE["ocr_conf_min"], name="ocr_conf_min"),
    ]

    @use_named_args(dims)
    def objective(**params):
        preprocess = {
            "width": params["width"],
            "blur_method": params["blur_method"],
            "gaussian_kernel": params["gaussian_kernel"],
            "bilateral_d": params["bilateral_d"],
            "sigma_color": params["sigma_color"],
            "sigma_space": params["sigma_space"],
            "gamma": params["gamma"],
            "clahe_clip": params["clahe_clip"],
            "clahe_tile_grid": params["clahe_tile_grid"],
            "block_size": params["adaptive_block"],
            "adaptive_block": params["adaptive_block"],
            "adaptive_C": params["adaptive_C"],
            "thresh_method": params["thresh_method"],
            "inv": params["inv"],
            "deskew_apply": params["deskew_apply"],
            "deskew_border": params["deskew_border"],
            "kernel_size": params["kernel_size"],
            "open_iter": params["open_iter"],
            "close_iter": params["close_iter"],
            "dilate_iter": params["dilate_iter"],
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
            crop_offsets=CROP_OFFSETS,
            conf=params["yolo_conf"],
            nms_iou=params["yolo_nms"],
            shrink_ratio=params["shrink_ratio"],
            tesseract_config=tess_cfg,
            ocr_conf_min=params["ocr_conf_min"],
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
            "width": best_vals["width"],
            "blur_method": best_vals["blur_method"],
            "gaussian_kernel": best_vals["gaussian_kernel"],
            "bilateral_d": best_vals["bilateral_d"],
            "sigma_color": best_vals["sigma_color"],
            "sigma_space": best_vals["sigma_space"],
            "gamma": best_vals["gamma"],
            "clahe_clip": best_vals["clahe_clip"],
            "clahe_tile_grid": best_vals["clahe_tile_grid"],
            "block_size": best_vals["adaptive_block"],
            "adaptive_block": best_vals["adaptive_block"],
            "adaptive_C": best_vals["adaptive_C"],
            "thresh_method": best_vals["thresh_method"],
            "inv": best_vals["inv"],
            "deskew_apply": best_vals["deskew_apply"],
            "deskew_border": best_vals["deskew_border"],
            "kernel_size": best_vals["kernel_size"],
            "open_iter": best_vals["open_iter"],
            "close_iter": best_vals["close_iter"],
            "dilate_iter": best_vals["dilate_iter"],
        },
        "crop_offsets": CROP_OFFSETS,
        "shrink_ratio": best_vals["shrink_ratio"],
        "conf": best_vals["yolo_conf"],
        "nms": best_vals["yolo_nms"],
        "ocr_conf_min": best_vals["ocr_conf_min"],
        "tesseract": (
            f"--oem {best_vals['oem']} --psm {best_vals['psm']} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    }

    return best_params, best_acc


def optimize_from_config(num_threads: int = 8):
    """Wywołaj optymalizację korzystając z danych i ścieżek z config."""
    data = load_dataset(CSV_PATH)
    return find_best_parameters(
        data,
        IMAGES_DIR,
        num_threads=num_threads,
    )


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