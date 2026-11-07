"""
lenet_ref.py - batched LeNet-5 reference, float32 and ap_fixed<16,8>.

Mirrors the HLS kernel's loop structure exactly, including AP_TRN truncation
after every accumulation. Vectorized over images so 12,630 run in under a
minute instead of hours.

Imported by measure_accuracy.py. Can also be run directly to self-test.
"""

import re

import numpy as np

IMG_H, IMG_W = 32, 32
N_CLASSES = 43
SCALE = 256.0          # ap_fixed<16,8> -> 8 fractional bits


def parse_weights(path):
    """Read a C header of weight arrays into {name: ndarray}."""
    txt = open(path, "r", errors="replace").read()
    out = {}
    pat = re.compile(
        r'(?:const\s+)?(?:float|data_t)\s+(\w+)\s*((?:\[\d+\])+)\s*=\s*(.*?);',
        re.S)
    for m in pat.finditer(txt):
        name, dims, body = m.group(1), m.group(2), m.group(3)
        shape = tuple(int(d) for d in re.findall(r'\[(\d+)\]', dims))
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', body)
        arr = np.array([float(x) for x in nums], dtype=np.float64)
        if arr.size != int(np.prod(shape)):
            raise ValueError(f"{name}: parsed {arr.size}, expected {np.prod(shape)}")
        out[name] = arr.reshape(shape)
    need = ["conv1_weights", "conv1_bias", "conv2_weights", "conv2_bias",
            "fc1_weights", "fc1_bias", "fc2_weights", "fc2_bias",
            "fc3_weights", "fc3_bias"]
    missing = [n for n in need if n not in out]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return out


def Q(x):
    """ap_fixed<16,8>, AP_TRN: truncate toward negative infinity."""
    return np.floor(x * SCALE) / SCALE


def forward(P, X, fixed):
    """
    X : (N, 32, 32) float, already normalized to [-1, 1]
    returns (N, 43) scores

    The quantizer is applied after EVERY accumulation, which is what
    `data_t sum += a*b;` does in C++. Applying it only at the end would
    silently produce the floating-point answer.
    """
    q = Q if fixed else (lambda v: np.asarray(v, dtype=np.float32))
    X = q(np.asarray(X, dtype=np.float64))
    N = X.shape[0]

    # ---- conv1: valid 5x5, 1 -> 6; bias first, then 25 truncating MACs -----
    w1, b1 = P["conv1_weights"][:, 0], P["conv1_bias"]
    c1 = np.empty((N, 6, 28, 28))
    for co in range(6):
        s = np.full((N, 28, 28), b1[co], dtype=np.float64)
        for kh in range(5):
            for kw in range(5):
                s = q(s + X[:, kh:kh+28, kw:kw+28] * w1[co, kh, kw])
        c1[:, co] = s
    c1 = np.maximum(c1, 0)
    p1 = np.maximum.reduce([c1[:, :, 0::2, 0::2], c1[:, :, 0::2, 1::2],
                            c1[:, :, 1::2, 0::2], c1[:, :, 1::2, 1::2]])

    # ---- conv2: ps[kw] from zero over (ci, kh), then folded with bias -----
    w2, b2 = P["conv2_weights"], P["conv2_bias"]
    c2 = np.empty((N, 16, 10, 10))
    for co in range(16):
        ps = []
        for kw in range(5):
            p = np.zeros((N, 10, 10))
            for ci in range(6):
                for kh in range(5):
                    p = q(p + p1[:, ci, kh:kh+10, kw:kw+10] * w2[co, ci, kh, kw])
            ps.append(p)
        s = np.full((N, 10, 10), b2[co], dtype=np.float64)
        for kw in range(5):
            s = q(s + ps[kw])
        c2[:, co] = s
    c2 = np.maximum(c2, 0)
    p2 = np.maximum.reduce([c2[:, :, 0::2, 0::2], c2[:, :, 0::2, 1::2],
                            c2[:, :, 1::2, 0::2], c2[:, :, 1::2, 1::2]])

    V = p2.reshape(N, 400)            # (co, ph, pw), the trained flatten order

    def dense(Vin, W, b, relu):
        O = np.tile(np.asarray(b, dtype=np.float64), (Vin.shape[0], 1))
        for j in range(Vin.shape[1]):
            O = q(O + Vin[:, j:j+1] * W[:, j])
        return np.maximum(O, 0) if relu else O

    H1 = dense(V,  P["fc1_weights"], P["fc1_bias"], True)
    H2 = dense(H1, P["fc2_weights"], P["fc2_bias"], True)
    return dense(H2, P["fc3_weights"], P["fc3_bias"], False)


def predict(P, X, fixed, chunk=512, progress=None):
    """forward() in chunks, so memory stays bounded for large sets."""
    N = len(X)
    out = np.empty((N, N_CLASSES))
    for i in range(0, N, chunk):
        out[i:i+chunk] = forward(P, X[i:i+chunk], fixed)
        if progress:
            progress(min(i + chunk, N), N)
    return out


CLASSES = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing >3.5t',
    11:'Right-of-way at next intersection', 12:'Priority road', 13:'Yield',
    14:'Stop', 15:'No vehicles', 16:'Vehicles >3.5t prohibited', 17:'No entry',
    18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right',
    21:'Double curve', 22:'Bumpy road', 23:'Slippery road',
    24:'Road narrows right', 25:'Road work', 26:'Traffic signals',
    27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing',
    30:'Beware of ice/snow', 31:'Wild animals crossing', 32:'End of all limits',
    33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
    36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing',
    42:'End of no passing >3.5t'}