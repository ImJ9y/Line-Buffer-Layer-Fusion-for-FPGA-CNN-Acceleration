# Line-Buffer Layer Fusion for FPGA CNN Acceleration

## Quick Links

- **Source Code DOI:** https://doi.org/10.5281/zenodo.22065607
- **GitHub Repository:** https://github.com/ImJ9y/Line-Buffer-Layer-Fusion-for-FPGA-CNN-Acceleration
- **Paper:** [Link when available]


HLS source and measurement data for a LeNet-5 traffic-sign accelerator on the
Xilinx PYNQ-Z2 (xc7z020-clg400-1), comparing three buffering architectures.

The proposed design fuses each convolution layer with its following max-pooling
layer: the convolution is reordered to produce the four results one pooling
window needs, they are reduced immediately, and the intermediate feature map is
never formed. Because this reorders the arithmetic rather than reducing it, every
computed value is bit-identical to the unfused reference.

---

## Results

All figures from Vitis HLS 2025.2 and Vivado 2025.2, `ap_fixed<16,8>` weights,
50 MHz implementation clock.

### Effect of fusion (buffering scheme held constant)

| Metric | Line buffer, no fusion | Fused line buffer | Effect |
|---|---:|---:|---|
| Cycles (initiation interval) | 13,096 | 3,978 | **3.29× faster** |
| LUT | 91,700 | 36,288 | **2.53× smaller** |
| Flip-flops | 81,256 | 27,543 | **2.95× smaller** |
| BRAM_18K | 85 | 85 | unchanged |
| DSP | 140 | 195 | 1.39× more |

Fusion is faster *and* smaller. Without it, the convolution and pooling stages
each need a completely partitioned band; fusion collapses them into one stage
reading one band, so the second partitioned array and the multiplexer network
addressing it are never instantiated. Block RAM is unchanged, which shows the
intermediate feature map is eliminated rather than relocated into memory. The
cost is arithmetic: four convolution windows per pooled output raises DSP usage
from 140 to 195 slices.

### All three architectures, both weight formats (HLS estimates)

| Architecture | Weights | Cycles | LUT | FF | DSP | BRAM |
|---|---|---:|---:|---:|---:|---:|
| Line buffer, no fusion | 32-bit float | 13,596 | 135,463 | 107,877 | 140 | 347 |
| Line buffer, no fusion | `ap_fixed<16,8>` | 13,096 | 91,700 | 81,256 | 140 | 85 |
| Non-line buffer, no fusion | 32-bit float | 10,837 | 143,089 | 86,279 | 200 | 355 |
| Non-line buffer, no fusion | `ap_fixed<16,8>` | 10,582 | 20,915 | 21,680 | 200 | 93 |
| **Fused line buffer** | 32-bit float | 4,034 | 95,407 | 60,761 | 195 | 377 |
| **Fused line buffer** | `ap_fixed<16,8>` | **3,978** | **36,288** | **27,543** | **195** | **85** |

The device has 53,200 LUTs. No float build fits, and the line-buffer design
without fusion needs 172% of the device even after quantization.

### Post-implementation (Vivado, accelerator hierarchy only)

| Metric | Non-line buffer | Fused | Ratio |
|---|---:|---:|---|
| LUT | 9,505 | 23,794 | 2.50× |
| Flip-flops | 13,229 | 21,072 | 1.59× |
| F7 multiplexers | 556 | 1,685 | 3.03× |
| BRAM tiles (36 Kb) | 35 | 55 | 1.57× |
| DSP | 204 | 208 | 1.02× |
| Worst negative slack | 12.494 ns | 12.430 ns | — |
| Critical path | 7.506 ns | 7.570 ns | — |
| **Maximum frequency** | **133.2 MHz** | **132.1 MHz** | — |

**Fusion is timing-neutral.** The critical path differs by 0.9% across a design
2.50× larger in LUTs, so the throughput advantage costs nothing in achievable
frequency. Interface logic (AXI DMA, interconnect, SmartConnect) occupies 3,952
LUTs and is identical in both implementations, so all differences above are
attributable to the accelerator.

### On hardware

Both bitstreams run on the PYNQ-Z2 and reproduce the reference classification on
every inference tested — one, four and seven inferences within a single
invocation, and twenty-one across three successive invocations.

| | Non-line buffer | Fused |
|---|---:|---:|
| Sustained throughput (140 images) | 1,272 img/s | 1,959 img/s |
| Marginal cost per image | 649.1 µs | 400.5 µs |
| Fixed cost per DMA transaction | 851 µs | 769 µs |

Measured throughput improves 1.54× rather than 2.66×, because a per-image
transfer cost of roughly 400 µs exceeds the computation time of either
accelerator. The AXI DMA's default 14-bit buffer-length register caps a single
transfer at 16,383 bytes, or seven images.

---

## Repository layout

```
src/
  FINAL_fused.cpp             proposed fused line-buffer design
  FINAL_ONE_LINE_BUFFER.cpp   line buffer, no fusion (does not fit the device)
  FINAL_NON_LINE_BUFFER.cpp   two-array buffering, no fusion
  lenet.h                     network dimensions and data types
  parameters.h                trained weights, 32-bit float
  parameters_fixed.h          trained weights, ap_fixed<16,8>
  image_data.h                one 32x32 test image
testbench/
  tb_lenet.cpp                C simulation testbench
scripts/
  lenet_board_test_final.py   board verification and timing measurement
```

Each `.cpp` is a complete, self-contained top level. Only one is added to a
project at a time.

---

## Building

### High-level synthesis

```bash
vitis-run --mode hls --csynth --config hls_config.cfg --work_dir <name>
```

`hls_config.cfg`:

```ini
part=xc7z020clg400-1
flow_target=vivado
package.output.format=ip_catalog
syn.file=FINAL_fused.cpp
syn.file=lenet.h
syn.file=parameters_fixed.h
tb.file=tb_lenet.cpp
syn.top=lenet_predict
```

Clock constraint 10 ns with 7.3 ns uncertainty.

> **Windows path length.** Vivado rejects paths over 260 characters, and the
> generated ROM filenames are long. If IP export fails with `[Common 17-680]`,
> map a short drive first:
> ```
> subst X: C:\path\to\project
> ```

### Vivado block design

The accelerator exposes two AXI-Stream ports of **different widths**, set by the
types in `lenet.h`:

| Port | Type | Width | DMA channel |
|---|---|---|---|
| `input_r` | `ap_fixed<16,8>` | **16 bit** | `M_AXIS_MM2S` |
| `output_r` | `ap_axis<32,0,0,0>` | **32 bit** | `S_AXIS_S2MM` |

Getting these wrong is the most common integration mistake: the DMA defaults to
32 bits on both channels, and leaving MM2S at 32 silently feeds two pixels per
beat.

**Zynq7 Processing System**

- `PS-PL Configuration` → `M AXI GP0` enabled (AXI-Lite control)
- `PS-PL Configuration` → `S AXI HP0` enabled (DMA access to DDR)
- `Clock Configuration` → `FCLK_CLK0` = **50 MHz**

**AXI Direct Memory Access**

- Enable Scatter Gather Engine: **unchecked** (simple mode)
- Width of Buffer Length Register: **14** — this is what caps a single
  transfer at 2^14 − 1 = 16,383 bytes, or seven images at 2,048 bytes each.
  Raise it to 26 if you want larger transactions.
- Address Width: 32
- **Read Channel (MM2S)**: enabled, Memory Map Data Width 64,
  **Stream Data Width 16**, Max Burst Size 16
- **Write Channel (S2MM)**: enabled, Memory Map Data Width 64,
  **Stream Data Width 32**, Max Burst Size 16

**Connections**

| From | To | Via |
|---|---|---|
| `processing_system7_0/M_AXI_GP0` | `axi_dma_0/S_AXI_LITE`, `lenet_predict_0/s_axi_control` | AXI SmartConnect |
| `axi_dma_0/M_AXI_MM2S`, `M_AXI_S2MM` | `processing_system7_0/S_AXI_HP0` | AXI Interconnect |
| `axi_dma_0/M_AXIS_MM2S` | `lenet_predict_0/input_r` | direct |
| `lenet_predict_0/output_r` | `axi_dma_0/S_AXIS_S2MM` | direct |

Both stream connections are required. Leaving either unconnected ties
`input_r_TVALID` low, and the accelerator blocks forever on its first read while
Vivado reports only a warning:

```
[BD 41-759] ... /lenet_predict_0/input_r_TVALID ... tied-off to all 0's
[xilinx.com:ip:axi_dma:7.1-11] S_AXIS_S2MM interface is unconnected
```

Drive every `aclk` from `FCLK_CLK0` and every `aresetn` from the Processor
System Reset block, then run `Validate Design` (F6) and
`Address Editor` → `Assign All`.

### Running on the board

```python
from pynq import Overlay, allocate
import numpy as np

ol  = Overlay("fused.bit")          # .hwh must sit beside it
dma, ip = ol.axi_dma_0, ol.lenet_predict_0

q = np.clip(np.round(img * 256), -32768, 32767).astype(np.int16)   # ap_fixed<16,8>

tx = allocate(shape=(len(q),), dtype=np.int16)
rx = allocate(shape=(1,),      dtype=np.int32)
tx[:] = q

ip.register_map.batches = 1
ip.register_map.CTRL.AP_START = 1
dma.recvchannel.transfer(rx)
dma.sendchannel.transfer(tx)
dma.sendchannel.wait(); dma.recvchannel.wait()
print(int(rx[0]))
```

At most **seven images** fit in one DMA transaction, from the 16,383-byte
buffer-length limit above. Split larger batches across invocations.

This snippet runs a single inference as a sanity check. It does not reproduce
the measurements in the paper — those come from a batch sweep and a linear
fit:

```python
import numpy as np, time

def run(img_q, n):                      # n <= 7
    tx = allocate(shape=(n * 1024,), dtype=np.int16)
    rx = allocate(shape=(n,),        dtype=np.int32)
    try:
        tx[:] = np.tile(img_q, n)
        ip.register_map.batches = n
        ip.register_map.CTRL.AP_START = 1
        t0 = time.perf_counter()
        dma.recvchannel.transfer(rx); dma.sendchannel.transfer(tx)
        dma.sendchannel.wait();       dma.recvchannel.wait()
        return [int(v) for v in rx], time.perf_counter() - t0
    finally:
        tx.freebuffer(); rx.freebuffer()

b = np.array([1, 2, 4, 7], float)
t = np.array([run(img_q, int(n))[1] for n in b])
marginal, fixed = np.polyfit(b, t, 1)
print("fixed %.0f us | marginal %.1f us" % (fixed*1e6, marginal*1e6))
```

The **marginal** cost is the per-image latency reported in the paper; the
**fixed** cost is host driver and DMA descriptor setup and is the same for both
designs. Sustained throughput comes from 140 images issued back to back in
chunks of seven.

---

## Design notes

**Parallelism.** The four convolution windows of a pooling block are independent,
so `SLIDE_PAR` sets how many are evaluated concurrently and `GROUPS` how many
sequentially, with their product always four. The shipped configuration uses
`SLIDE_PAR_C1 = 2` and `SLIDE_PAR_C2 = 4`, giving 50 and 120 DSP slices. Raising
the first stage to 4 would need about 100 slices and bring the total to 245,
exceeding the 220 available.

**Band depth.** The fused line buffer holds K+1 rows rather than K. For a pooled
output at (0,0) with K=5 and stride 2, the upper windows span input rows 0–4 and
the lower span 1–5, so six rows must be resident at once.

**Addressing.** Slide offsets derive from the loop counters as
`s = sg·P + u`, `dh = s >> 1`, `dw = s & 1`. Because `u` is unrolled and `sg` is a
loop counter, these are compile-time constants in each unrolled instance, so the
buffer row each MAC unit reads is fixed and no multiplexer is inferred across
the rows of the band.

**Array partitioning.** The partition factor must be a multiple of the channel
count. `conv2` indexes as `(w + kw)·C_in + c_i`, so a factor that is not a
multiple of `C_in` places successive channels in the same bank and reintroduces
the conflict the partitioning is meant to remove.

**fc1 tail masking.** The first fully connected layer produces 120 outputs and is
tiled sixteen wide, so the output loop covers 128 positions. In C simulation the
surplus is harmless because each call gets fresh streams, but in hardware the
FIFOs persist between invocations and the surplus misaligns the next inference.
All three designs guard the tail so only the 120 real outputs are written.

---

## Citing

If you use this work, please cite the paper:

```bibtex
@inproceedings{im2026linebuffer,
  author    = {Im, Jeonghun and Grover, Radhika S.},
  title     = {An {FPGA}-Based {CNN} Accelerator Architecture Using
               Line-Buffer Layer Fusion for {ADAS} Applications},
  year      = {2026},
  note      = {IEEE - Processing}
}
```

To cite the source code, use:

J. Im and R. S. Grover, "Line-Buffer Layer Fusion for FPGA CNN Acceleration (Source Code)," v1.0.0, Zenodo, Aug. 2026. https://doi.org/10.5281/zenodo.22065607

---

## Environment

AMD Vitis HLS 2025.2 · AMD Vivado 2025.2 · PYNQ-Z2, xc7z020-clg400-1 · PYNQ v3.x

## License

MIT. See `LICENSE`.
