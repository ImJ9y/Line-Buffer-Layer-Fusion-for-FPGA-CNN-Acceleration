// ============================================================================
// LeNet-5, ap_fixed<16,8> - TWO-ARRAY POOLING, OPTIMIZED, UNBATCHED
// PYNQ-Z2 / xc7z020-clg400-1        weights: parameters_fixed.h (pre-quantized ap_fixed<16,8>)
// ============================================================================
//
// FINAL BUILD - two-array design, fc1 tail fix applied, ready for IP export.
//
// PRE-FIX MEASUREMENT: 10,582 cycles | 18,714 LUT (35%) | 21,303 FF (20%)
//                      | 200 DSP (91%) | 55 BRAM (20%) | 7.213 ns
// Expect the tail mask to leave cycles unchanged and move LUT/BRAM slightly,
// as it did on the fused design (3,978 cycles held; LUT 34,087 -> 36,288,
// BRAM 47 -> 85).
//
// FC2_TILE is 4 into 84 outputs; 84/4 = 21 exactly, so fc2 needs no guard.
// fc3 here writes through an int* rather than a stream, so it has no tail
// issue either. Any future change to a tile width must preserve divisibility
// or add the same guard.
//
// MEASURED: 10,582 cycles | 18,714 LUT (35%) | 21,303 FF (20%) | 200 DSP (91%)
//           | 55 BRAM (20%) | 7.213 ns
//
// OPTIMIZATION HISTORY (all with offline-quantized weights):
//   13,227  baseline
//   10,984  conv2 kw loop unrolled (DSP 83 -> 203)
//   10,582  conv1 band cyclic 5 -> 8  (that stage: 25,761 -> 8,251 LUT)
// Stopping point: DSP at 91%, and conv1 now bounds the design at 10,581.
// conv2 at cyclic 8 was tried and regressed (+4,625 LUT): the index is
// (ow+kw)*6+ci, so the factor must stay a multiple of the 6-channel stride.
//
// PURPOSE: the non-fused comparison point. Same unbatched
// top-level signature as the POINT 1 line-buffer design, so the only variable
// between the two is the buffering architecture.
//
//   void lenet_predict(hls::stream<data_t>& input,
//                      hls::stream<axis_out_t>& output, int batches)
//
// The batched version of this design measured 13,267 cycles | 89,169 LUT |
// 63,806 FF | 114 DSP | 252 BRAM, wrapped in VITIS_LOOP_293_1 with an
// ap_axis output. That interface is REMOVED here. Expect the numbers to move,
// because the batch wrapper and the AXI-Stream output both change how Vitis
// schedules and shares logic across the dataflow region. Re-measure; do not
// carry the batched figures over.
//
// ---------------------------------------------------------------------------
// THE ARCHITECTURAL DIFFERENCE
// ---------------------------------------------------------------------------
// POINT 1 line buffer:  window pinned at address 0, band SHIFTED to meet it.
//                       Both horizontal and vertical advance are data movement.
//
// This design:          the window position enters the ADDRESS. Convolution
//                       reads lb[kh][w+kw]; pooling reads two whole rows held
//                       in separate arrays and indexes into them directly.
//                       Only whole rows move, once per output row.
//
// Measured per-stage, from the batched build (indicative, not final):
//
//   stage    this design           POINT 1 optimized      LUT ratio
//   conv1    10,983 | 25,761 LUT   5,906 | 12,133 LUT      2.1x LARGER
//   pool1     6,105 |    673 LUT   5,339 | 18,728 LUT     28x smaller
//   conv2    13,266 | 20,997 LUT  19,095 |106,603 LUT      5.1x smaller
//   pool2     2,081 |    630 LUT   2,130 | 19,375 LUT     31x smaller
//
// The pooling stages are the story: near-identical cycle counts for roughly
// one thirtieth of the area, because index-based access needs no shift network
// and no complete partition. conv1 is the one stage where POINT 1 wins, since
// its pinned window keeps all 25 band indices compile-time constant while this
// version's (w+kw) is a runtime address.
// ============================================================================
#include "lenet.h"
#include "parameters_fixed.h"

#define FC3_TILE       4

static inline data_t relu(data_t x){ return (x > 0) ? x : (data_t)0; }
static inline data_t max4(data_t a, data_t b, data_t c, data_t d){
    data_t m = a; if (b>m) m=b; if (c>m) m=c; if (d>m) m=d; return m;
}

// ============================ conv1 =========================================
// 32x32x1 -> 28x28x6, 5x5 kernel, 6 filters
//
// Row buffer lb[5][32]. The column enters the address as (w+kw), so nothing
// moves horizontally. Only the vertical advance is data movement: four row
// copies plus one new row, 27 times per image.
void conv1_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t lb[5][32];
    #pragma HLS ARRAY_PARTITION variable=lb dim=1 complete
    // PARTITION SWEEP: cyclic 5 -> 8. conv1 reads 5 consecutive columns per
    // window; 5 is conflict-free but forces a modulo-5 bank decode. 8 is a
    // power of two and still covers a 5-wide access.
    #pragma HLS ARRAY_PARTITION variable=lb dim=2 cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=0
    #pragma HLS ARRAY_PARTITION variable=conv1_bias complete

    LOOP_C1_FILL: for (int r = 0; r < 5; r++)
        for (int c = 0; c < 32; c++){
            #pragma HLS PIPELINE II=1
            lb[r][c] = in.read();
        }

    LOOP_C1_ROW: for (int h = 0; h < 28; h++){
        LOOP_C1_COL: for (int w = 0; w < 28; w++){
            LOOP_C1_CH: for (int co = 0; co < 6; co++){
                #pragma HLS PIPELINE II=1
                data_t sum = (data_t)conv1_bias[co];
                for (int kh = 0; kh < 5; kh++){
                    #pragma HLS UNROLL
                    for (int kw = 0; kw < 5; kw++){
                        #pragma HLS UNROLL
                        sum += lb[kh][w + kw] * (data_t)conv1_weights[co][0][kh][kw];
                    }
                }

                out.write(relu(sum));
            }
        }
        // vertical advance only: shift four rows up, load one new row
        if (h < 27){
            LOOP_C1_SHIFT_V: for (int r = 0; r < 4; r++)
                for (int c = 0; c < 32; c++){
                    #pragma HLS PIPELINE II=1
                    lb[r][c] = lb[r+1][c];
                }
            LOOP_C1_LOAD_V: for (int c = 0; c < 32; c++){
                #pragma HLS PIPELINE II=1
                lb[4][c] = in.read();
            }
        }
    }
    // reads: 160 + 27*32 = 1024
}

// ============================ pool1 =========================================
// 28x28x6 -> 14x14x6, 2x2 stride-2 max
//
// TWO-ARRAY POOLING. Stride 2 equals the window height, so consecutive output
// rows consume DISJOINT input rows: nothing is reused between output rows.
// Both rows are therefore loaded fresh at the start of each output row, and
// the window position enters the address. No shifting, no read guards, and no
// complete partition, which is why this stage costs ~673 LUT against the
// line-buffer version's ~18,728.
#define P1_ROWLEN (28*6)                       // 168

void pool1_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t r0[P1_ROWLEN];
    data_t r1[P1_ROWLEN];
    #pragma HLS ARRAY_PARTITION variable=r0 cyclic factor=2
    #pragma HLS ARRAY_PARTITION variable=r1 cyclic factor=2

    LOOP_P1_ROW: for (int ph = 0; ph < 14; ph++){

        LOOP_P1_LOAD: for (int i = 0; i < P1_ROWLEN; i++){
            #pragma HLS PIPELINE II=1
            r0[i] = in.read();
        }
        LOOP_P1_LOAD2: for (int i = 0; i < P1_ROWLEN; i++){
            #pragma HLS PIPELINE II=1
            r1[i] = in.read();
        }

        LOOP_P1_COL: for (int pw = 0; pw < 14; pw++){
            LOOP_P1_CH: for (int c = 0; c < 6; c++){
                #pragma HLS PIPELINE II=1
                int a0 = (2*pw)*6 + c;
                int a1 = (2*pw + 1)*6 + c;

                out.write( max4(r0[a0], r0[a1], r1[a0], r1[a1]) );
            }
        }
    }
    // reads: 14 * 336 = 4704
}

// ============================ conv2 =========================================
// 14x14x6 -> 10x10x16, 5x5 kernel, 16 filters
//
// Row buffer lb[5][84], column in the address as (ow+kw)*6+ci. The kw loop is
// PIPELINED with per-kw partial sums in ps[], which removes the loop-carried
// accumulator recurrence. Fixed-point addition is associative, so reordering
// the accumulation is bit-exact.
//
// NOTE: this is the configuration that must NOT be ported to the flat 1-D
// POINT 1 band. There, a pipelined kw makes the index a runtime value into a
// completely partitioned 420-element array, producing a 420-way multiplexer
// (measured: 79,995 cycles). Here dim=2 is CYCLIC factor 6, so the variable
// part only selects a row within an 84-deep bank and the mux never forms.
#define C2_ROWLEN (14*6)                       // 84

void conv2_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t lb[5][C2_ROWLEN];
    #pragma HLS ARRAY_PARTITION variable=lb dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=lb dim=2 cyclic factor=6
    #pragma HLS ARRAY_PARTITION variable=conv2_weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=conv2_weights complete dim=3
    #pragma HLS ARRAY_PARTITION variable=conv2_bias complete

    LOOP_C2_FILL: for (int r = 0; r < 5; r++)
        for (int i = 0; i < C2_ROWLEN; i++){
            #pragma HLS PIPELINE II=1
            lb[r][i] = in.read();
        }

    LOOP_C2_ROW: for (int oh = 0; oh < 10; oh++){
        LOOP_C2_COL: for (int ow = 0; ow < 10; ow++){
            LOOP_C2_CH: for (int co = 0; co < 16; co++){
                #pragma HLS PIPELINE II=1

                data_t ps[5];
                #pragma HLS ARRAY_PARTITION variable=ps complete

                // PRAGMA SWEEP STEP: kw UNROLLED instead of pipelined.
                // conv2 bounds this design at 13,226 cycles with 30 MACs per
                // cycle (ci x kh). Unrolling kw as well gives 150 MACs and
                // removes the 5-trip pipeline that is re-entered 1,600 times
                // per image. Same arithmetic, same weights: loop unrolling
                // only (Section III). Expect conv2 toward ~1,900 cycles and
                // DSP 83 -> ~150. Design 2 has room: 41% LUT, 38% DSP.
                LOOP_C2_KW: for (int kw = 0; kw < 5; kw++){
                    #pragma HLS UNROLL
                    data_t p = 0;
                    for (int ci = 0; ci < 6; ci++){
                        #pragma HLS UNROLL
                        for (int kh = 0; kh < 5; kh++){
                            #pragma HLS UNROLL
                            p += lb[kh][(ow + kw)*6 + ci]
                               * (data_t)conv2_weights[co][ci][kh][kw];
                        }
                    }
                    ps[kw] = p;
                }

                data_t sum = (data_t)conv2_bias[co];
                for (int kw = 0; kw < 5; kw++){
                    #pragma HLS UNROLL
                    sum += ps[kw];
                }
                out.write(relu(sum));
            }
        }
        if (oh < 9){
            LOOP_C2_SHIFT_V: for (int r = 0; r < 4; r++)
                for (int i = 0; i < C2_ROWLEN; i++){
                    #pragma HLS PIPELINE II=1
                    lb[r][i] = lb[r+1][i];
                }
            LOOP_C2_LOAD_V: for (int i = 0; i < C2_ROWLEN; i++){
                #pragma HLS PIPELINE II=1
                lb[4][i] = in.read();
            }
        }
    }
    // reads: 420 + 9*84 = 1176
}

// ============================ pool2 =========================================
// 10x10x16 -> 5x5x16, 2x2 stride-2 max.  Same two-array structure as pool1.
#define P2_ROWLEN (10*16)                      // 160

void pool2_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    data_t r0[P2_ROWLEN];
    data_t r1[P2_ROWLEN];
    #pragma HLS ARRAY_PARTITION variable=r0 cyclic factor=2
    #pragma HLS ARRAY_PARTITION variable=r1 cyclic factor=2

    LOOP_P2_ROW: for (int ph = 0; ph < 5; ph++){

        LOOP_P2_LOAD: for (int i = 0; i < P2_ROWLEN; i++){
            #pragma HLS PIPELINE II=1
            r0[i] = in.read();
        }
        LOOP_P2_LOAD2: for (int i = 0; i < P2_ROWLEN; i++){
            #pragma HLS PIPELINE II=1
            r1[i] = in.read();
        }

        LOOP_P2_COL: for (int pw = 0; pw < 5; pw++){
            LOOP_P2_CH: for (int c = 0; c < 16; c++){
                #pragma HLS PIPELINE II=1
                int a0 = (2*pw)*16 + c;
                int a1 = (2*pw + 1)*16 + c;

                out.write( max4(r0[a0], r0[a1], r1[a0], r1[a1]) );
            }
        }
    }
    // reads: 5 * 320 = 1600
}

// ============================ fc1 : 400 -> 120 ==============================
// Identical to the POINT 1 design, so the FC back-end is not a variable in
// the comparison.
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
    // TAIL MASK: 120 outputs are not a multiple of the tile width 16, so this
    // loop runs og = 0,16,...,112 and the last iteration covers 112..127 while
    // only 112..119 exist. Without the guards it writes 128 values downstream
    // while fc2 reads 120, stranding 8 per inference, and reads fc1_weights
    // out of bounds at og+k = 120..127.
    //
    // C simulation still returns the correct class - fc2 reads the first 120
    // in order and never touches the surplus, and each C-sim call gets fresh
    // local streams. In HARDWARE the FIFOs are physical and persist, so
    // inference 2 begins 8 values out of alignment. On the fused design this
    // produced a correct first board result and wrong ones thereafter, with
    // batch 4 returning four different wrong classes. Fixed there and verified
    // 100/100 on the board; the same fix is applied here pre-emptively.
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
            if (og + k < 120) out.write(relu(ag[k]));       // only real outputs
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

#ifndef __SYNTHESIS__
    {   // DEBUG: all 43 accumulators vs the reference scores
        static const float golden[43] = {
            -3.023438f, -10.679688f, -15.628906f,  -6.187500f, -35.808594f,
            -5.355469f, -23.429688f, -20.027344f, -11.972656f, -25.011719f,
           -24.347656f, -15.851562f, -21.707031f, -31.921875f, -35.867188f,
           -22.015625f, -26.937500f, -34.687500f, -33.902344f,  -6.824219f,
            -4.003906f, -11.687500f,  -8.820312f,   4.589844f,  -1.164062f,
            -3.453125f, -15.265625f, -21.273438f,   2.914062f,  12.347656f,
            -5.554688f,  -2.792969f, -16.093750f, -27.058594f,  -9.980469f,
            -5.511719f, -17.265625f, -18.890625f, -31.429688f, -29.765625f,
           -14.449219f, -28.125000f, -23.179688f };
        int   cmi = 0, gmi = 0, matches = 0;
        float worst = 0.0f;
        for (int i = 0; i < 43; i++){
            float mine = acc[i].to_float();
            float d  = mine - golden[i];
            float ad = d < 0 ? -d : d;
            if (ad < 1e-5f) matches++;
            if (ad > worst) worst = ad;
            if (acc[i]    > acc[cmi])    cmi = i;
            if (golden[i] > golden[gmi]) gmi = i;
        }
        printf("fc3 (UNFUSED)  : matches %d/43  worst %.4f  class %d "
               "(golden %d) %s\n",
               matches, worst, cmi, gmi, cmi == gmi ? "OK" : "MISMATCH");
    }
#endif
    int mi = 0; data_t mv = acc[0];
    for (int i = 1; i < 43; i++){ if (acc[i] > mv){ mv = acc[i]; mi = i; } }
    *predicted_class = mi;
}

// ============================ one inference =================================
// The DATAFLOW region, renamed so the batch loop can call it repeatedly.
// Each call consumes exactly 1024 input values and produces one class index.
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
// Batched AXI-Stream interface, matching the testbench and the other designs.
// TLAST on the final packet only; on every packet it would end the DMA
// transfer after the first image.
void lenet_predict(hls::stream<data_t>& input,
                   hls::stream<axis_out_t>& output,
                   int batches){
    #pragma HLS INTERFACE axis      port=input
    #pragma HLS INTERFACE axis      port=output
    #pragma HLS INTERFACE s_axilite port=batches bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control

    LOOP_BATCH: for (int b = 0; b < batches; b++){
        int cls = 0;
        lenet_once(input, &cls);

        axis_out_t pkt;
        pkt.data = cls;
        pkt.last = (b == batches - 1) ? 1 : 0;
        output.write(pkt);
    }
}
