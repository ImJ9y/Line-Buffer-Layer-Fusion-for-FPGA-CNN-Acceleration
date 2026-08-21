# =============================================================================
# LeNet-5 PYNQ-Z2 board test  --  final version, both variants
#
# Purpose:
#   1. Determine whether a bitstream is PRE-FIX or POST-FIX (fc1 tail mask)
#      by running the SAME image several times and checking the outputs agree.
#   2. Confirm the predicted class matches the golden reference (29).
#   3. Measure classification accuracy over a test set (optional).
#   4. Measure wall-clock throughput and separate it from per-call overhead.
#   5. Write everything to JSON so the board data lives with the build.
#
# DMA LIMIT: the AXI DMA was generated with a 14-bit Buffer Length Register,
# capping one transfer at 16,383 bytes. At 1024 int16 values per image that is
# 2,048 bytes per image, so MAX_BATCH = 7. Anything larger is split across
# multiple invocations by run_chunked(). To lift this, set the DMA's "Width of
# Buffer Length Register" to 26 in Vivado and rebuild -- but rebuild BOTH
# variants together so the infrastructure baseline stays comparable.
#
# Notebook-safe: no sys.argv. Paste cell by cell (split on "# %%") or run the
# whole file as a script.
#
# Run once per variant:
#     BITSTREAM = "mixed_fused.bit"  -> cells 1-8 -> record
#     BITSTREAM = "two_array.bit"    -> cells 1-8 -> record
# Loading a second overlay tears down the first, so finish one completely
# before switching.
#
# Files needed on the board:
#   <variant>.bit and <variant>.hwh   (same base name, same directory)
#   image_data.h                      (found automatically, cell 3)
#   test_images.npy / test_labels.npy (optional, cell 6 skips without them)
# =============================================================================


# %% cell 1 -- load the overlay -----------------------------------------------
from pynq import Overlay, allocate
import numpy as np
import time
import json
import os
import re
import glob

BITSTREAM = "mixed_fused.bit"     # <-- change to "two_array.bit" for run 2
VARIANT   = os.path.splitext(os.path.basename(BITSTREAM))[0]

ol = Overlay(BITSTREAM)
print("bitstream :", BITSTREAM)
print("IPs       :", list(ol.ip_dict.keys()))

assert 'axi_dma_0'       in ol.ip_dict, ol.ip_dict.keys()
assert 'lenet_predict_0' in ol.ip_dict, ol.ip_dict.keys()

dma = ol.axi_dma_0
ip  = ol.lenet_predict_0

print()
print(ip.register_map)


# %% cell 2 -- pre-flight checks ----------------------------------------------
# AP_IDLE should read 1 on a freshly loaded, reset accelerator.
# 'batches' reading write-only is normal for an s_axilite input scalar.

USE_REGISTER_MAP = True
CTRL_OFF, BATCH_OFF = 0x00, 0x10      # fallback, from xlenet_predict_hw.h

try:
    idle = int(ip.register_map.CTRL.AP_IDLE)
    print("AP_IDLE   :", idle, "(expect 1)")
    assert idle == 1, "accelerator not idle -- reload the overlay"
except (AttributeError, KeyError) as e:
    USE_REGISTER_MAP = False
    print("register_map unavailable (%s) -- using raw offsets" % e)
    print("AP_IDLE   :", (ip.read(CTRL_OFF) >> 2) & 1, "(expect 1)")

print("mode      :", "register_map" if USE_REGISTER_MAP else "raw offsets")

# Read the DMA's actual limit rather than assuming it, so this script keeps
# working if the Buffer Length Register is widened later.
try:
    DMA_MAX_BYTES = int(dma.sendchannel._max_size)
except AttributeError:
    DMA_MAX_BYTES = 16383
print("DMA max   : %d bytes" % DMA_MAX_BYTES)


# %% cell 3 -- build test_image.npy from image_data.h -------------------------
IMG_LEN      = 1024      # 32 x 32
FRAC_BITS    = 8         # ap_fixed<16,8> -> 8 fractional bits
GOLDEN_CLASS = 29        # argmax of golden[] in fc3_stream's debug block

MAX_BATCH = DMA_MAX_BYTES // (IMG_LEN * 2)
print("MAX_BATCH : %d images per invocation (%d bytes each)"
      % (MAX_BATCH, IMG_LEN * 2))

if os.path.exists('test_image.npy'):
    print("test_image.npy already present -- skipping parse")
else:
    cand = (['image_data.h', 'jay/image_data.h', '../image_data.h']
            + glob.glob('**/image_data.h', recursive=True))
    HEADER = next((p for p in cand if os.path.exists(p)), None)
    if HEADER is None:
        raise FileNotFoundError("image_data.h not found; set HEADER by hand")
    print("using", HEADER)

    text = open(HEADER).read()
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)    # block comments
    text = re.sub(r"//[^\n]*", " ", text)                 # line comments

    braces = re.findall(r"\{(.*?)\}", text, flags=re.S)
    if not braces:
        raise ValueError("no brace-enclosed initializer found")
    body = max(braces, key=len)        # longest block is the pixel array

    tokens = re.findall(r"[-+]?(?:0[xX][0-9a-fA-F]+|\d*\.\d+(?:[eE][-+]?\d+)?"
                        r"|\d+\.?\d*(?:[eE][-+]?\d+)?)[fFuUlL]*", body)
    vals = []
    for t in tokens:
        t = t.rstrip("fFuUlL")
        vals.append(float(int(t, 16))
                    if t.lower().lstrip("+-").startswith("0x") else float(t))

    arr = np.asarray(vals, dtype=np.float32)
    print("parsed %d values" % arr.size)

    if arr.size > IMG_LEN and arr.size % IMG_LEN == 0:
        print("looks like %d images; taking the first" % (arr.size // IMG_LEN))
        arr = arr[:IMG_LEN]
    assert arr.size == IMG_LEN, "expected %d, got %d" % (IMG_LEN, arr.size)

    if np.all(arr == np.round(arr)) and np.abs(arr).max() > 16:
        print("raw int16 detected -- dividing by 2^%d" % FRAC_BITS)
        arr = arr / float(1 << FRAC_BITS)

    print("range %.4f .. %.4f   mean %.4f"
          % (arr.min(), arr.max(), arr.mean()))
    np.save('test_image.npy', arr)
    print("wrote test_image.npy")

_a = np.load('test_image.npy').reshape(32, 32)
print()
print('\n'.join(''.join(' .:-=+*#%@'[int(np.clip((v + 1) / 2, 0, 1) * 9)]
                        for v in row) for row in _a))


# %% cell 4 -- helpers --------------------------------------------------------
def quantize(img_float):
    """float -> ap_fixed<16,8> raw int16, matching make_fixed_params.py."""
    q = np.round(np.asarray(img_float, dtype=np.float32) * (1 << FRAC_BITS))
    return np.clip(q, -32768, 32767).astype(np.int16)


def start_accel(batches):
    if USE_REGISTER_MAP:
        ip.register_map.batches = batches
        ip.register_map.CTRL.AP_START = 1
    else:
        ip.write(BATCH_OFF, batches)
        ip.write(CTRL_OFF, 0x01)


def run(img_q_flat, batches):
    """
    One DMA invocation. batches must be <= MAX_BATCH.
    returns (list of predicted classes, elapsed seconds)

    Elapsed covers DMA setup, transfer, compute and readback -- the full
    system path, so it will exceed the initiation-interval projection.
    Buffers are freed in `finally`; a leak here eventually starves the board
    of contiguous memory.
    """
    assert img_q_flat.size == batches * IMG_LEN, \
        "expected %d values, got %d" % (batches * IMG_LEN, img_q_flat.size)
    assert batches <= MAX_BATCH, \
        "batches=%d exceeds DMA limit of %d -- use run_chunked()" \
        % (batches, MAX_BATCH)

    tx = allocate(shape=(batches * IMG_LEN,), dtype=np.int16)
    rx = allocate(shape=(batches,),           dtype=np.int32)
    try:
        tx[:] = img_q_flat
        start_accel(batches)

        t0 = time.perf_counter()
        dma.recvchannel.transfer(rx)
        dma.sendchannel.transfer(tx)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        t1 = time.perf_counter()

        return [int(v) for v in rx], (t1 - t0)
    finally:
        tx.freebuffer()
        rx.freebuffer()


def run_chunked(img_q_flat, total, chunk=None):
    """Split `total` images into invocations of at most MAX_BATCH."""
    chunk = chunk or MAX_BATCH
    preds, elapsed = [], 0.0
    for i in range(0, total, chunk):
        n = min(chunk, total - i)
        p, dt = run(img_q_flat[i*IMG_LEN:(i+n)*IMG_LEN], n)
        preds.extend(p)
        elapsed += dt
    return preds, elapsed


def tile(img_q, batches):
    """Repeat one quantized image `batches` times, end to end."""
    return np.tile(img_q, batches)


# %% cell 5 -- fix-state test (the decisive one) ------------------------------
# Feed the SAME image repeatedly under ONE ap_start. Identical input must give
# identical output. Any drift is FIFO misalignment, not a data artifact.
#
#   [29, 29, 29, 29]  -> POST-FIX
#   [29, 14,  3, 22]  -> PRE-FIX  (first correct, then 8 stranded values per
#                                  inference shift the stream alignment)

img   = np.load('test_image.npy').reshape(-1)
img_q = quantize(img)
print("quantized range  : %d .. %d  (expect -256 .. 256 for +/-1.0)"
      % (img_q.min(), img_q.max()))

# Control: batches=1 twice. PRE-FIX PASSES this, because each fresh ap_start
# sees empty FIFOs. If this fails, the problem is quantization or wiring, not
# the tail mask, and the batch results below would only mislead.
single_a, _ = run(img_q, 1)
single_b, _ = run(img_q, 1)
print("batches=1 (x2)   :", single_a, single_b)

r4, _ = run(tile(img_q, 4), 4)
fix_state = "POST-FIX" if len(set(r4)) == 1 else "PRE-FIX"
print("batches=4        :", r4, "->", fix_state)

# Deepest single-invocation test the DMA allows. Misalignment compounds across
# all MAX_BATCH inferences, so this is the strongest available signal.
rmax, _ = run(tile(img_q, MAX_BATCH), MAX_BATCH)
print("batches=%-9d:" % MAX_BATCH, rmax)
print("  unique classes :", len(set(rmax)), "(1 = post-fix)")

# Cross-invocation check: FIFOs persist between ap_start pulses, so a residue
# bug would surface here even if it hid within a single invocation.
rc, _ = run_chunked(tile(img_q, 21), 21)
print("chunked x21      :", len(set(rc)), "unique ->",
      "consistent" if len(set(rc)) == 1 else "DRIFT ACROSS INVOCATIONS")

if len(set(r4)) == 1 and r4[0] != GOLDEN_CLASS:
    print()
    print("  WARNING: consistent but class %d != golden %d." % (r4[0], GOLDEN_CLASS))
    print("           Consistency still proves POST-FIX. The class mismatch is")
    print("           a separate quantization question -- check FRAC_BITS")
    print("           against data_t's fractional width.")


# %% cell 6 -- accuracy sweep (skipped if dataset absent) ---------------------
acc = None
wall_ips = None

if os.path.exists('test_images.npy') and os.path.exists('test_labels.npy'):
    imgs   = np.load('test_images.npy').reshape(-1, IMG_LEN)
    labels = np.load('test_labels.npy').reshape(-1)
    N = len(imgs)
    print("test set         : %d images, %d invocations of <=%d"
          % (N, -(-N // MAX_BATCH), MAX_BATCH))

    preds, dt = run_chunked(quantize(imgs).reshape(-1), N)
    acc      = float(np.mean(np.array(preds) == labels))
    wall_ips = N / dt

    print("accuracy         : %.4f  (%d/%d)" % (acc, int(round(acc * N)), N))
    print("wall-clock       : %.1f img/s over %.3f s" % (wall_ips, dt))

    wrong = np.where(np.array(preds) != labels)[0]
    if len(wrong):
        print("first 10 misses  :",
              [(int(i), int(labels[i]), int(preds[i])) for i in wrong[:10]])
else:
    print("test_images.npy / test_labels.npy not found -- skipping")


# %% cell 7 -- throughput decomposition ---------------------------------------
# Per-call overhead (register writes, DMA descriptor setup, Python) amortizes
# as batch size grows, but MAX_BATCH caps how far it can amortize. The plateau
# is therefore DMA-limited, not accelerator-limited -- report both it and the
# II projection, and say which constraint is binding.

sizes = [b for b in (1, 2, 4, 8, 16, 32) if b <= MAX_BATCH] + [MAX_BATCH]
sizes = sorted(set(sizes))

print()
print("%8s %12s %12s   %s" % ("batches", "seconds", "img/s", "mode"))
sweep = {}
for b in sizes:
    _, dt = run(tile(img_q, b), b)
    sweep[b] = b / dt
    print("%8d %12.6f %12.1f   single" % (b, dt, sweep[b]))

# Sustained rate across many back-to-back invocations -- the realistic figure.
TOTAL = MAX_BATCH * 20
_, dt_sus = run_chunked(tile(img_q, TOTAL), TOTAL)
sustained = TOTAL / dt_sus
print("%8d %12.6f %12.1f   chunked" % (TOTAL, dt_sus, sustained))

plateau = max(list(sweep.values()) + [sustained])
print()
print("plateau          : %.1f img/s" % plateau)
print("sustained        : %.1f img/s (%d images, chunked)" % (sustained, TOTAL))
print("per-call overhead: %.3f ms (from batches=1)" % (1000.0 / sweep[1]))

II_CYCLES = {"two_array": 10582, "mixed_fused": 3978}
CLK_HZ    = 50e6
theo = None
if VARIANT in II_CYCLES:
    theo = CLK_HZ / II_CYCLES[VARIANT]
    print("II projection    : %.1f img/s  (%d cyc @ %.0f MHz)"
          % (theo, II_CYCLES[VARIANT], CLK_HZ / 1e6))
    print("achieved         : %.1f%% of projection" % (100 * sustained / theo))
    print("               -> the shortfall is DMA + host overhead, not the")
    print("                  accelerator; MAX_BATCH=%d caps amortization."
          % MAX_BATCH)


# %% cell 8 -- write results --------------------------------------------------
results = {
    "variant":              VARIANT,
    "bitstream":            BITSTREAM,
    "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
    "clock_mhz":            CLK_HZ / 1e6,
    "dma_max_bytes":        DMA_MAX_BYTES,
    "max_batch":            MAX_BATCH,
    "batches_1_repeated":   [single_a, single_b],
    "batches_4":            r4,
    "batches_max":          rmax,
    "chunked_21_unique":    len(set(rc)),
    "fix_state":            fix_state,
    "golden_class":         GOLDEN_CLASS,
    "class_matches_golden": bool(len(set(r4)) == 1 and r4[0] == GOLDEN_CLASS),
    "accuracy":             acc,
    "wall_clock_img_s":     wall_ips,
    "throughput_sweep":     {str(k): v for k, v in sweep.items()},
    "sustained_img_s":      sustained,
    "plateau_img_s":        plateau,
    "ii_projection_img_s":  theo,
}

out = "board_results_%s.json" % VARIANT
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print()
print("=" * 62)
print("VARIANT    :", VARIANT)
print("FIX STATE  :", fix_state)
print("CLASS      :", r4[0] if len(set(r4)) == 1 else "inconsistent",
      "(golden %d)" % GOLDEN_CLASS)
print("ACCURACY   :", "n/a" if acc is None else "%.4f" % acc)
print("SUSTAINED  : %.1f img/s" % sustained)
print("II PROJ    :", "n/a" if theo is None else "%.1f img/s" % theo)
print("=" * 62)
print("wrote", out)
print()
print("Next: set BITSTREAM to the other variant, re-run cells 1-8.")
print("Then copy board_results_*.json into results/<variant>/ alongside the")
print(".dcp, post-route .rpt files, csynth.rpt and the .cpp.")
