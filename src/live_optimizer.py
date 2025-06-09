"""Optimize OCR preprocessing parameters using live camera feed.

This module captures frames from a PC webcam and searches for the best
parameters that allow recognition of a provided ground-truth license
plate. Results are stored in JSON for later reuse.
"""

from __future__ import annotations

import ast
import json
from typing import List

import cv2
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

from config import (
    CAMERA_INDEX,
    LIVE_CROP_OFFSETS,
    LIVE_SHRINK_RATIO,
    OPTIMIZER_SETTINGS,
    OPTIMIZER_SPACE,
)
from detector import PlateDetector
from ocr import recognize_plate
from utils import crop_bbox


# ---------------------------------------------------------------------------
# Helpers copied from ``optimization.py`` to encode tuple parameters
# ---------------------------------------------------------------------------

def _encode_tuple(t: tuple[int, int]) -> str:
    """Return a stable string representation for a tuple dimension."""
    return repr(t)


def _decode_tuple(val):
    """Decode tuple string back to tuple of ints."""
    if isinstance(val, str) and val.startswith("("):
        return tuple(ast.literal_eval(val))
    return val


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def _capture_frames(num_frames: int) -> List[cv2.MatLike]:
    """Capture ``num_frames`` frames from the configured webcam."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    frames: List[cv2.MatLike] = []
    try:
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            # small delay so camera buffer refreshes
            cv2.waitKey(30)
    finally:
        cap.release()
    return frames


def _evaluate_frames(
    detector: PlateDetector,
    frames: List[cv2.MatLike],
    gt_plate: str,
    *,
    preprocess_params: dict | None = None,
    crop_offsets: dict | None = None,
    conf: float | None = None,
    nms_iou: float | None = None,
    shrink_ratio: float = 0.0,
    tesseract_config: str | None = None,
    ocr_conf_min: float = 0.0,
) -> float:
    """Return accuracy of recognizing ``gt_plate`` on given frames."""
    if crop_offsets is None:
        crop_offsets = {"x1": 0, "x2": 0, "y1": 0, "y2": 0}
    correct = 0

    for idx, frame in enumerate(frames):
        dets = detector.detect(frame, conf=conf, iou=nms_iou)
        if len(dets) == 0:
            continue
        bbox = dets[0][:4]
        plate_img = crop_bbox(frame, bbox, offsets=crop_offsets, shrink_ratio=shrink_ratio)
        pred_text, _ = recognize_plate(
            plate_img,
            f"frame{idx}",
            preprocess_params=preprocess_params,
            tesseract_config=tesseract_config,
            ocr_conf_min=ocr_conf_min,
        )
        if pred_text == gt_plate:
            correct += 1
    return correct / len(frames) if frames else 0.0


# ---------------------------------------------------------------------------
# Optimization routine
# ---------------------------------------------------------------------------

def optimize_from_camera(
    gt_plate: str,
    *,
    num_frames: int = 20,
    n_calls: int | None = None,
    output_file: str = "live_best_params.json",
) -> tuple[dict, float]:
    """Search for parameters using frames captured from webcam."""
    detector = PlateDetector()
    frames = _capture_frames(num_frames)

    if n_calls is None:
        n_calls = OPTIMIZER_SETTINGS.get("n_calls", 50)
    random_state = OPTIMIZER_SETTINGS.get("random_state", 42)
    n_random = OPTIMIZER_SETTINGS.get("n_random_starts", 10)

    dims = [
        Real(*OPTIMIZER_SPACE["yolo_conf"], name="yolo_conf"),
        Real(*OPTIMIZER_SPACE["yolo_nms"], name="yolo_nms"),
        Real(*OPTIMIZER_SPACE["shrink_ratio"], name="shrink_ratio"),
        Categorical(OPTIMIZER_SPACE["deskew_apply"], name="deskew_apply"),
        Categorical(OPTIMIZER_SPACE["deskew_border"], name="deskew_border"),
        Integer(*OPTIMIZER_SPACE["width"], name="width"),
        Real(*OPTIMIZER_SPACE["clahe_clip"], name="clahe_clip"),
        Categorical([
            _encode_tuple(t) for t in OPTIMIZER_SPACE["clahe_tile_grid"]
        ], name="clahe_tile_grid"),
        Real(*OPTIMIZER_SPACE["gamma"], name="gamma"),
        Categorical(OPTIMIZER_SPACE["blur_method"], name="blur_method"),
        Categorical([
            _encode_tuple(t) for t in OPTIMIZER_SPACE["gaussian_kernel"]
        ], name="gaussian_kernel"),
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
        params["gaussian_kernel"] = _decode_tuple(params["gaussian_kernel"])
        params["clahe_tile_grid"] = _decode_tuple(params["clahe_tile_grid"])
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

        acc = _evaluate_frames(
            detector,
            frames,
            gt_plate,
            preprocess_params=preprocess,
            crop_offsets=LIVE_CROP_OFFSETS,
            conf=params["yolo_conf"],
            nms_iou=params["yolo_nms"],
            shrink_ratio=params["shrink_ratio"],
            tesseract_config=tess_cfg,
            ocr_conf_min=params["ocr_conf_min"],
        )
        # gp_minimize minimizes, so return negative accuracy
        return -acc

    result = gp_minimize(
        func=objective,
        dimensions=dims,
        acq_func="EI",
        n_calls=n_calls,
        n_random_starts=n_random,
        random_state=random_state,
    )

    best_vals = dict(zip((d.name for d in dims), result.x))
    best_vals["gaussian_kernel"] = _decode_tuple(best_vals["gaussian_kernel"])
    best_vals["clahe_tile_grid"] = _decode_tuple(best_vals["clahe_tile_grid"])
    best_acc = -result.fun

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
        "crop_offsets": LIVE_CROP_OFFSETS,
        "shrink_ratio": best_vals["shrink_ratio"],
        "conf": best_vals["yolo_conf"],
        "nms": best_vals["yolo_nms"],
        "ocr_conf_min": best_vals["ocr_conf_min"],
        "tesseract": (
            f"--oem {best_vals['oem']} --psm {best_vals['psm']} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": best_acc, "params": best_params}, f, indent=2)

    return best_params, best_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize parameters using live camera feed")
    parser.add_argument("--frames", type=int, default=150, help="number of frames to capture")
    parser.add_argument("--calls", type=int, default=None, help="optimizer iterations")
    parser.add_argument("--output", default="live_best_params.json", help="where to store found parameters")
    args = parser.parse_args()

    params, acc = optimize_from_camera(
        gt_plate="GKW5KE2",
        num_frames=args.frames,
        n_calls=args.calls,
        output_file=args.output,
    )
    print("Best accuracy:", acc)
    print("Params:", params)