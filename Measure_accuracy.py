#!/usr/bin/env python3
"""
make_testset.py - build gtsrb_test.bin for tb_accuracy.cpp

Downloads the GTSRB test split, preprocesses it exactly as the accelerator
expects, and writes one binary file. Vitis does the measuring.

    py -m pip install kagglehub pillow numpy pandas
    py make_testset.py

Preprocessing, matching image_data.h and the reference PYNQ script:
    Image.open(p).convert('L').resize((32, 32))
    (arr - 127.5) / 127.5                        -> [-1, 1]

Output format, which tb_accuracy.cpp reads:
    int32   count, height(32), width(32)
    float32 pixels[count][32][32]    row-major
    int32   labels[count]            0..42
"""

import os
import struct
import sys

import numpy as np
import pandas as pd
from PIL import Image

import kagglehub

# --- settings; Windows paths need the r"..." prefix ------------------------
DATASET = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
OUT_BIN = r"C:\Users\jeong\TEST_FINAL_Research\FINAL_fusion\gtsrb_test.bin"


def find_test_csv(root):
    """Test.csv holds the held-out labels: 12,630 rows of Path and ClassId."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == "test.csv":
                return os.path.join(dirpath, fn)
    return None


def main():
    root = kagglehub.dataset_download(DATASET)
    csv_path = find_test_csv(root)
    if not csv_path:
        sys.exit(f"No Test.csv under {root}")

    base = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "classid" not in cols or "path" not in cols:
        sys.exit(f"{csv_path} has no ClassId/Path columns")
    print(f"{csv_path}\n{len(df)} images")

    X = np.empty((len(df), 32, 32), dtype=np.float32)
    y = np.empty(len(df), dtype=np.int32)

    for i, (_, r) in enumerate(df.iterrows()):
        p = os.path.join(base, str(r[cols["path"]]).replace("\\", "/"))
        img = Image.open(p).convert("L").resize((32, 32))
        X[i] = (np.array(img, dtype=np.float32) - 127.5) / 127.5
        y[i] = int(r[cols["classid"]])
        if (i + 1) % 2000 == 0:
            print(f"  {i+1} / {len(df)}")

    lo, hi = float(X.min()), float(X.max())
    print(f"\npixel range : {lo:.4f} to {hi:.4f}")
    print(f"labels      : {int(y.min())} to {int(y.max())}")
    if lo < -1.05 or hi > 1.05:
        sys.exit("STOP: range outside [-1, 1]; the preprocessing is wrong.")
    if y.min() < 0 or y.max() > 42:
        sys.exit("STOP: labels must be 0..42")

    with open(OUT_BIN, "wb") as f:
        f.write(struct.pack("<iii", len(X), 32, 32))
        f.write(np.ascontiguousarray(X, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(y, dtype="<i4").tobytes())

    print(f"\nwrote {OUT_BIN}  ({os.path.getsize(OUT_BIN)/1e6:.1f} MB)")
    print("\nIn Vitis, point TEST_FILE in tb_accuracy.cpp at that path with")
    print("forward slashes, then pick the precision in the design:")
    print("  ap_fixed  ->  lenet.h: typedef ap_fixed<16,8> data_t;")
    print("                FINAL_fused.cpp: #include \"parameters_fixed.h\"")
    print("  float32   ->  lenet.h: typedef float data_t;")
    print("                FINAL_fused.cpp: #include \"parameters.h\"")
    print("and run C Simulation.")


if __name__ == "__main__":
    main()