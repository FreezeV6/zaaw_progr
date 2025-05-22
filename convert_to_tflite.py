#!/usr/bin/env python3
import argparse
import subprocess
import tensorflow as tf
import os

def onnx_to_saved_model(onnx_path, out_dir):
    print(f"🚀 ONNX → SavedModel: {onnx_path} → {out_dir}")
    # wywołujemy CLI onnx-tf
    subprocess.run([
        "onnx-tf", "convert",
        "-i", onnx_path,
        "-o", out_dir
    ], check=True)
    print("✅ SavedModel gotowy.")

def saved_model_to_tflite(saved_model_dir, tflite_path):
    print(f"🚀 SavedModel → TFLite: {saved_model_dir} → {tflite_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("✅ TFLite zapisany jako", tflite_path)

def main():
    p = argparse.ArgumentParser("ONNX→SavedModel→TFLite")
    p.add_argument("--onnx",       required=True, help="ścieżka do .onnx")
    p.add_argument("--saved_model",required=True, help="katalog SavedModel")
    p.add_argument("--tflite",     required=True, help="ścieżka wyjściowa .tflite")
    args = p.parse_args()

    onnx_to_saved_model(args.onnx,       args.saved_model)
    saved_model_to_tflite(args.saved_model, args.tflite)

if __name__ == "__main__":
    main()
