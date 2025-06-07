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
        Integer(*OPTIMIZER_SPACE["width"], name="width"),
        Categorical(OPTIMIZER_SPACE["bilateral_d"], name="bilateral_d"),
        Categorical(OPTIMIZER_SPACE["block_size"], name="block_size"),
        Integer(*OPTIMIZER_SPACE["c"], name="c"),
        Categorical(OPTIMIZER_SPACE["inv"], name="inv"),
        Real(*OPTIMIZER_SPACE["conf"], name="conf"),
        Integer(*OPTIMIZER_SPACE["x1"], name="x1"),
        Integer(*OPTIMIZER_SPACE["x2"], name="x2"),
        Integer(*OPTIMIZER_SPACE["y1"], name="y1"),
        Integer(*OPTIMIZER_SPACE["y2"], name="y2"),
        Categorical(OPTIMIZER_SPACE["thresh_method"], name="thresh_method"),
        Categorical(OPTIMIZER_SPACE["deskew_apply"], name="deskew_apply"),
        Categorical(OPTIMIZER_SPACE["psm"], name="psm"),
        Categorical(OPTIMIZER_SPACE["oem"], name="oem"),
    ]

    @use_named_args(dims)
    def objective(**params):
        preprocess = {
            "width":         params["width"],
            "bilateral_d":   params["bilateral_d"],
            "block_size":    params["block_size"],
            "c":             params["c"],
            "thresh_method": params["thresh_method"],
            "inv":           params["inv"],
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
            "width": best_vals["width"],
            "bilateral_d": best_vals["bilateral_d"],
            "block_size": best_vals["block_size"],
            "c": best_vals["c"],
            "thresh_method": best_vals["thresh_method"],
            "inv": best_vals["inv"],
            "deskew_apply": best_vals["deskew_apply"],
        },
        "crop_offsets": {
            "x1": best_vals["x1"],
            "x2": best_vals["x2"],
            "y1": best_vals["y1"],
            "y2": best_vals["y2"],
        },
        "conf": best_vals["conf"],
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