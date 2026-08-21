// ============================================================================
// LeNet-5, ap_fixed<16,8> - 1-D LINE-BUFFER (DESIGN 1), OPTIMIZED, BATCHED
// MATCHED-TIER BUILD: conv2 flattening from the fit pass + FC1_TILE 16 so the
// FC back-end matches design2 and design3 exactly.
// PYNQ-Z2 / xc7z020-clg400-1        weights: parameters_fixed.h (pre-quantized ap_fixed<16,8>)
// ============================================================================
//
// BATCHED counterpart to one_line_buffer_point1_optimized_best.cpp. The seven
// stage functions are BYTE-IDENTICAL to the unbatched version; only the top
// level changed. Interface now matches the two-array batched build:
//
//   void lenet_predict(hls::stream<data_t>& input,
//                      hls::stream<axis_out_t>& output,
//                      int batches)
//
// so the same testbench drives both.
//
// ---------------------------------------------------------------------------
// EXPECTED RESULT
// ---------------------------------------------------------------------------
// On the two-array design, batching changed nothing measurable:
//
//   metric      batched     unbatched    delta
//   interval     13,267      13,267        0
//   LUT          89,169      89,578      +0.5%
//   FF           63,806      63,963      +0.2%
//   DSP             114         114        0
//   BRAM            252         256      +1.6%
//
// The reported interval is PER IMAGE in both cases. Batching amortises the
// ~0.4-0.5 ms Python/DMA driver overhead, which is invisible to HLS synthesis
// and only shows up in on-board wall-clock measurement. Expect POINT 1 to
// behave the same way: roughly 19,096 cycles per image, ~198,000 LUT, 291 DSP.
// If it does not, something in the wrapper is interacting with the dataflow
// region and that is itself worth reporting.
//
// ---------------------------------------------------------------------------
// UNBATCHED REFERENCE NUMBERS FOR THIS DESIGN
// ---------------------------------------------------------------------------
//   total     19,096 cycles | 198,051 LUT (372%) | 147,939 FF | 291 DSP
//                           | 218 BRAM | 6.943 ns
//
//   (numbers below predate the conv2 flattening; that change measured
//    13,596 cycles | 124,340 LUT | 132 DSP at FC1_TILE 8)
//   stage      interval    LUT      DSP    II    at FIFO port floor
//   conv1        5,906    12,133     85     6    yes (6 writes/col)
//   pool1        5,339    18,728      0    12    yes (12 reads/col)
//   conv2       19,095   106,603    150     -    no, 11.9x above floor
//   pool2        2,130    19,375      0    32    yes (32 reads/col)
//   fc1          6,864     9,710      9
//   fc2          3,191     5,009      4
//   fc3            238    25,673     43
//
// Vitis labels the conv1/pool1/pool2 limits "Resource Limitation": their II
// equals the number of hls::stream accesses per iteration, which is a hard
// floor no pragma can move.
// ============================================================================
#include "lenet.h"
#include "parameters.h"

#define FC3_TILE       4

static inline data_t relu(data_t x){ return (x > 0) ? x : (data_t)0; }
static inline data_t max4(data_t a, data_t b, data_t c, data_t d){
    data_t m = a; if (b>m) m=b; if (c>m) m=c; if (d>m) m=d; return m;
}

// ============================ conv1 =========================================
// 32x32x1 -> 28x28x6, 5x5 kernel, 6 filters
#define C1_ROWLEN 32
#define C1_KS     5
#define C1_BAND   (C1_KS * C1_ROWLEN)          // 160
#define C1_AT(n,c) lb[(n) * C1_ROWLEN + (c)]
#define C1_SH_H   1
#define C1_SH_V   5

void conv1_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t lb[C1_BAND];
    #pragma HLS ARRAY_PARTITION variable=lb complete
    #pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=0
    #pragma HLS ARRAY_PARTITION variable=conv1_bias complete

    LOOP_C1_FILL: for (int i = 0; i < C1_BAND; i++){
        #pragma HLS PIPELINE II=1
        lb[i] = in.read();
    }

    LOOP_C1_ROW: for (int h = 0; h < 28; h++){

        // PIPELINE here, not on LOOP_C1_CH. Achieved II=6 = FIFO writes/col.
        LOOP_C1_COL: for (int w = 0; w < 28; w++){
            #pragma HLS PIPELINE

            LOOP_C1_CH: for (int co = 0; co < 6; co++){
                #pragma HLS UNROLL
                data_t sum = (data_t)conv1_bias[co];
                for (int kh = 0; kh < C1_KS; kh++){
                    #pragma HLS UNROLL
                    for (int kw = 0; kw < C1_KS; kw++){
                        #pragma HLS UNROLL
                        // kh, kw unrolled -> index is a compile-time constant,
                        // so each band read is a wire rather than a mux.
                        sum += C1_AT(kh, kw) * (data_t)conv1_weights[co][0][kh][kw];
                    }
                }
                out.write(relu(sum));
            }

            if (w < 27){
                LOOP_C1_SHIFT_H: for (int i = 0; i < C1_BAND - C1_SH_H; i++){
                    #pragma HLS UNROLL
                    lb[i] = lb[i + C1_SH_H];
                }
                if (h < 27) lb[C1_BAND - 1] = in.read();
            }
        }

        if (h < 27){
            LOOP_C1_SHIFT_V: for (int i = 0; i < C1_BAND - C1_SH_V; i++){
                #pragma HLS UNROLL
                lb[i] = lb[i + C1_SH_V];
            }
            LOOP_C1_LOAD_V: for (int i = 0; i < C1_SH_V; i++){
                #pragma HLS PIPELINE II=1
                lb[C1_BAND - C1_SH_V + i] = in.read();
            }
        }
    }
    // reads: 160 + 729 + 135 = 1024
}

// ============================ pool1 =========================================
// 28x28x6 -> 14x14x6, 2x2 stride-2 max
#define P1_ROWLEN (28*6)                       // 168
#define P1_BAND   (2 * P1_ROWLEN)              // 336
#define P1_AT(n,c) buf[(n) * P1_ROWLEN + (c)]
#define P1_SH_H   (2*6)                        // 12
#define P1_SH_V   180

void pool1_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t buf[P1_BAND];
    #pragma HLS ARRAY_PARTITION variable=buf complete

    LOOP_P1_FILL: for (int i = 0; i < P1_BAND; i++){
        #pragma HLS PIPELINE II=1
        buf[i] = in.read();
    }

    LOOP_P1_ROW: for (int ph = 0; ph < 14; ph++){

        // One pipeline at column level lets the max, the shift and the refill
        // share iterations. Achieved II=12 = FIFO reads/col.
        LOOP_P1_COL: for (int pw = 0; pw < 14; pw++){
            #pragma HLS PIPELINE

            LOOP_P1_CH: for (int c = 0; c < 6; c++){
                #pragma HLS UNROLL
                out.write( max4(P1_AT(0, c), P1_AT(0, 6 + c),
                                P1_AT(1, c), P1_AT(1, 6 + c)) );
            }

            if (pw < 13){
                LOOP_P1_SHIFT_H: for (int i = 0; i < P1_BAND - P1_SH_H; i++){
                    #pragma HLS PIPELINE II=1
                    buf[i] = buf[i + P1_SH_H];
                }
                if (ph < 13){
                    LOOP_P1_LOAD_H: for (int i = 0; i < P1_SH_H; i++){
                        #pragma HLS UNROLL
                        buf[P1_BAND - P1_SH_H + i] = in.read();
                    }
                }
            }
        }

        if (ph < 13){
            LOOP_P1_SHIFT_V: for (int i = 0; i < P1_BAND - P1_SH_V; i++){
                #pragma HLS UNROLL
                buf[i] = buf[i + P1_SH_V];
            }
            LOOP_P1_LOAD_V: for (int i = 0; i < P1_SH_V; i++){
                #pragma HLS PIPELINE II=1
                buf[P1_BAND - P1_SH_V + i] = in.read();
            }
        }
    }
    // reads: 336 + 2028 + 2340 = 4704
}

// ============================ conv2 =========================================
// 14x14x6 -> 10x10x16, 5x5 kernel, 16 filters
//
// Best of five measured configurations. Do NOT pipeline the kw loop and do NOT
// bank the band: under ARRAY_PARTITION complete, a pipelined kw makes the
// index a runtime value into 420 registers, producing a 420-way multiplexer
// (measured 79,995 cycles). Keeping every index constant forces the 150-wide
// unroll, which is the 150 DSP.
#define C2_ROWLEN (14*6)                       // 84
#define C2_KS     5
#define C2_BAND   (C2_KS * C2_ROWLEN)          // 420
#define C2_AT(n,c) lb[(n) * C2_ROWLEN + (c)]
#define C2_SH_H   6
#define C2_SH_V   30

void conv2_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t lb[C2_BAND];
    #pragma HLS ARRAY_PARTITION variable=lb complete
    #pragma HLS ARRAY_PARTITION variable=conv2_weights complete dim=3
    #pragma HLS ARRAY_PARTITION variable=conv2_weights complete dim=4
    #pragma HLS ARRAY_PARTITION variable=conv2_bias complete

    LOOP_C2_FILL: for (int i = 0; i < C2_BAND; i++){
        #pragma HLS PIPELINE II=1
        lb[i] = in.read();
    }

    LOOP_C2_ROW: for (int oh = 0; oh < 10; oh++){
        LOOP_C2_COL: for (int ow = 0; ow < 10; ow++){

            // FIT PASS 1: co and kw flattened into ONE 80-trip pipelined
            // loop with per-kw partial sums. Drops 150 MACs/cycle to 30,
            // which is where conv2's 106,603 LUT (57% of this design) comes
            // from. The flattening avoids the drain pathology that made an
            // earlier 5-trip kw pipeline cost 79,995 cycles: 100 loop entries
            // instead of 1,600. Bit-exact (fixed-point add is associative).
            data_t ps[C2_KS];
            #pragma HLS ARRAY_PARTITION variable=ps complete
            int co = 0, kw = 0;                    // counters, no divider

            LOOP_C2_CHKW: for (int t = 0; t < 16 * C2_KS; t++){
                #pragma HLS PIPELINE II=1
                data_t p = 0;
                for (int kh = 0; kh < C2_KS; kh++){
                    #pragma HLS UNROLL
                    for (int ci = 0; ci < 6; ci++){
                        #pragma HLS UNROLL
                        p += C2_AT(kh, kw * 6 + ci)
                           * (data_t)conv2_weights[co][ci][kh][kw];
                    }
                }
                ps[kw] = p;

                if (kw == C2_KS - 1){
                    data_t sum = (data_t)conv2_bias[co];
                    for (int k = 0; k < C2_KS; k++){
                        #pragma HLS UNROLL
                        sum += ps[k];
                    }
                    out.write(relu(sum));
                }
                if (kw == C2_KS - 1){ kw = 0; co++; } else { kw++; }
            }

            if (ow < 9){
                LOOP_C2_SHIFT_H: for (int i = 0; i < C2_BAND - C2_SH_H; i++){
                    #pragma HLS UNROLL
                    lb[i] = lb[i + C2_SH_H];
                }
                if (oh < 9){
                    LOOP_C2_LOAD_H: for (int i = 0; i < C2_SH_H; i++){
                        #pragma HLS UNROLL
                        lb[C2_BAND - C2_SH_H + i] = in.read();
                    }
                }
            }
        }

        if (oh < 9){
            LOOP_C2_SHIFT_V: for (int i = 0; i < C2_BAND - C2_SH_V; i++){
                #pragma HLS UNROLL
                lb[i] = lb[i + C2_SH_V];
            }
            LOOP_C2_LOAD_V: for (int i = 0; i < C2_SH_V; i++){
                #pragma HLS PIPELINE II=1
                lb[C2_BAND - C2_SH_V + i] = in.read();
            }
        }
    }
    // reads: 420 + 486 + 270 = 1176
}

// ============================ pool2 =========================================
// 10x10x16 -> 5x5x16, 2x2 stride-2 max
#define P2_ROWLEN (10*16)                      // 160
#define P2_BAND   (2 * P2_ROWLEN)              // 320
#define P2_AT(n,c) buf[(n) * P2_ROWLEN + (c)]
#define P2_SH_H   (2*16)                       // 32
#define P2_SH_V   192

void pool2_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t buf[P2_BAND];
    #pragma HLS ARRAY_PARTITION variable=buf complete

    LOOP_P2_FILL: for (int i = 0; i < P2_BAND; i++){
        #pragma HLS PIPELINE II=1
        buf[i] = in.read();
    }

    LOOP_P2_ROW: for (int ph = 0; ph < 5; ph++){

        // Achieved II=32 = FIFO reads/col.
        LOOP_P2_COL: for (int pw = 0; pw < 5; pw++){
            #pragma HLS PIPELINE

            LOOP_P2_CH: for (int c = 0; c < 16; c++){
                #pragma HLS UNROLL
                out.write( max4(P2_AT(0, c), P2_AT(0, 16 + c),
                                P2_AT(1, c), P2_AT(1, 16 + c)) );
            }

            if (pw < 4){
                LOOP_P2_SHIFT_H: for (int i = 0; i < P2_BAND - P2_SH_H; i++){
                    #pragma HLS PIPELINE II=1
                    buf[i] = buf[i + P2_SH_H];
                }
                if (ph < 4){
                    LOOP_P2_LOAD_H: for (int i = 0; i < P2_SH_H; i++){
                        #pragma HLS UNROLL
                        buf[P2_BAND - P2_SH_H + i] = in.read();
                    }
                }
            }
        }

        if (ph < 4){
            LOOP_P2_SHIFT_V: for (int i = 0; i < P2_BAND - P2_SH_V; i++){
                #pragma HLS UNROLL
                buf[i] = buf[i + P2_SH_V];
            }
            LOOP_P2_LOAD_V: for (int i = 0; i < P2_SH_V; i++){
                #pragma HLS PIPELINE II=1
                buf[P2_BAND - P2_SH_V + i] = in.read();
            }
        }
    }
    // reads: 320 + 512 + 768 = 1600
}

// ============================ fc1 : 400 -> 120 ==============================
void fc1_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    #pragma HLS ARRAY_PARTITION variable=fc1_weights dim=1 cyclic factor=16
    data_t xb[400];
    LOOP_FC1_LOAD: for (int i = 0; i < 400; i++){
        #pragma HLS PIPELINE II=1
        data_t v = in.read();
        int c = i % 16;
        int w = (i / 16) % 5;
        int h = (i / 80);
        xb[c*25 + h*5 + w] = v;                 // de-shuffle to flatten order
    }
    LOOP_FC1_OUT: for (int og = 0; og < 120; og += 16){
        data_t ag[16];
        #pragma HLS ARRAY_PARTITION variable=ag complete
        for (int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            ag[k] = (og + k < 120) ? (data_t)fc1_bias[og+k] : (data_t)0;
        }
        for (int j = 0; j < 400; j++){
            #pragma HLS PIPELINE II=1
            data_t v = xb[j];
            for (int k = 0; k < 16; k++){
                #pragma HLS UNROLL
                int oi = (og + k < 120) ? (og + k) : 119;   // clamp, in bounds
                data_t contrib = v * (data_t)fc1_weights[oi][j];
                if (og + k < 120) ag[k] += contrib;
            }
        }
        for (int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            if (og + k < 120) out.write(relu(ag[k]));   // only real outputs
        }
    }
}

// ============================ fc2 : 120 -> 84 ===============================
void fc2_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    #pragma HLS ARRAY_PARTITION variable=fc2_weights dim=1 cyclic factor=4
    data_t xb[120];
    LOOP_FC2_LOAD: for (int i = 0; i < 120; i++){
        #pragma HLS PIPELINE II=1
        xb[i] = in.read();
    }
    LOOP_FC2_OUT: for (int og = 0; og < 84; og += 4){
        data_t ag[4];
        #pragma HLS ARRAY_PARTITION variable=ag complete
        for (int k = 0; k < 4; k++){
            #pragma HLS UNROLL
            ag[k] = (data_t)fc2_bias[og+k];
        }
        for (int j = 0; j < 120; j++){
            #pragma HLS PIPELINE II=1
            data_t v = xb[j];
            for (int k = 0; k < 4; k++){
                #pragma HLS UNROLL
                ag[k] += v * (data_t)fc2_weights[og+k][j];
            }
        }
        for (int k = 0; k < 4; k++) out.write(relu(ag[k]));
    }
}

// ============================ fc3 : 84 -> 43 + argmax =======================
void fc3_stream(hls::stream<data_t>& in, int* predicted_class){
    // AREA FIX: output-tiled like fc1/fc2, replacing a 43-wide unroll that
    // cost ~25,700 LUT (23% of the design) for a stage finishing in 239
    // cycles. 11 groups x 84 inputs ~ 924 cycles, still far below any stage
    // bound. 43 is not a multiple of 4, so the last group's unused lane is
    // masked and its weight index clamped to stay in bounds.
    #pragma HLS ARRAY_PARTITION variable=fc3_weights dim=1 cyclic factor=FC3_TILE

    data_t xb[84];
    LOOP_FC3_BUF: for (int i = 0; i < 84; i++){
        #pragma HLS PIPELINE II=1
        xb[i] = in.read();
    }

    data_t acc[43];
    LOOP_FC3_OG: for (int og = 0; og < 43; og += FC3_TILE){
        data_t ag[FC3_TILE];
        #pragma HLS ARRAY_PARTITION variable=ag complete
        for (int k = 0; k < FC3_TILE; k++){
            #pragma HLS UNROLL
            ag[k] = (og + k < 43) ? (data_t)fc3_bias[og+k] : (data_t)0;
        }
        LOOP_FC3_J: for (int j = 0; j < 84; j++){
            #pragma HLS PIPELINE II=1
            data_t v = xb[j];
            for (int k = 0; k < FC3_TILE; k++){
                #pragma HLS UNROLL
                int oi = (og + k < 43) ? (og + k) : 42;
                data_t contrib = v * (data_t)fc3_weights[oi][j];
                if (og + k < 43) ag[k] += contrib;
            }
        }
        for (int k = 0; k < FC3_TILE; k++){
            #pragma HLS UNROLL
            if (og + k < 43) acc[og+k] = ag[k];
        }
    }
    int mi = 0; data_t mv = acc[0];
    for (int i = 1; i < 43; i++){ if (acc[i] > mv){ mv = acc[i]; mi = i; } }
    *predicted_class = mi;
}

// ============================ one inference =================================
// The DATAFLOW region. Identical to the unbatched top level, just renamed so
// the batch loop can call it repeatedly. Each call consumes exactly 1024 input
// values and produces one class index.
static void lenet_once(hls::stream<data_t>& input, int* predicted_class){
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    hls::stream<data_t> s1, s2, s3, s4, s5, s6;
    #pragma HLS STREAM variable=s1 depth=64
    #pragma HLS STREAM variable=s2 depth=64
    #pragma HLS STREAM variable=s3 depth=64
    #pragma HLS STREAM variable=s4 depth=64
    #pragma HLS STREAM variable=s5 depth=64
    #pragma HLS STREAM variable=s6 depth=64

    conv1_stream(input, s1);
    pool1_stream(s1, s2);
    conv2_stream(s2, s3);
    pool2_stream(s3, s4);
    fc1_stream(s4, s5);
    fc2_stream(s5, s6);
    fc3_stream(s6, predicted_class);
}

// ============================ top : batched =================================
// One ap_start and one DMA transfer for the whole batch. TLAST is asserted on
// the final packet only; without it the S2MM DMA hangs on hardware.
void lenet_predict(hls::stream<data_t>& input,
                   hls::stream<axis_out_t>& output,
                   int batches){
    #pragma HLS INTERFACE axis port=input
    #pragma HLS INTERFACE axis port=output
    #pragma HLS INTERFACE s_axilite port=batches bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control

    LOOP_BATCH: for (int b = 0; b < batches; b++){
        int cls = 0;
        lenet_once(input, &cls);

        axis_out_t pkt;
        pkt.data = cls;
        pkt.last = (b == batches - 1) ? 1 : 0;
        pkt.keep = -1;                 // all bytes valid
        pkt.strb = -1;
        output.write(pkt);
    }
}
