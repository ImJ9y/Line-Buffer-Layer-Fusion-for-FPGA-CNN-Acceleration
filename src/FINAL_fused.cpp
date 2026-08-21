// ============================================================================
// LeNet-5 (GTSRB, 43 classes) - FOUR-WINDOW-SLIDE FUSED ACCELERATOR
// ap_fixed<16,8>, PYNQ-Z2 / xc7z020-clg400-1
// ============================================================================
// VARIANT: MIXED  (conv1 lb[192] complete, conv2 lb[6][84] cyclic 12) - BEST
// conv1 stays flat: 36 of 192 elements per cycle makes the complete-partition
// mux affordable. conv2 is the exception: 72 of 504 per cycle needs a
// two-level partition that only the 2-D form can express.
// VERIFIED BUILD - ready for IP export.
//
// FUNCTIONAL STATUS: this design verifies 43/43 exact against the reference
// scores (worst delta 0.0000, predicted class 29) using tb_lenet_diagnostic.
// An earlier apparent failure - conv1 activations collapsing to max 0.281 -
// was traced to the OLD testbench streaming corrupted data, not to the design.
// With a correct testbench the fused stages match the unfused two-array design
// to six decimals: conv1+pool1 max 4.644531 mean 0.709898, conv2+pool2 max
// 16.292969 mean 1.922764. Two independent architectures agreeing to that
// precision confirms both.
//
// MEASURED: 3,978 cycles | 34,087 LUT (64%) | 27,166 FF (26%) | 195 DSP (89%)
//           | 47 BRAM (17%) | 7.213 ns.  6.84x -> 9.70x over the 38,590
//           reference.  Bound: conv1+pool1 at 3,977; conv2+pool2 at 3,640.
//
// STOPPING POINT (Dr. Grover's criterion): DSP at 89%. Raising SLIDE_PAR_C1
// from 2 to 4 would add ~50 DSP and exceed the 220 budget, and conv2 is no
// longer the bound, so cycles do not fall further without leaving the device.
//
// PRAGMA SWEEP STEP: SLIDE_PAR_C2 2 -> 4 (loop unrolling, Section III).
// conv2+pool2 bounds this design at 5,640 cycles; at 4 all four of Fig. 3's
// f1 units exist in the same cycle instead of two. Expect conv2 toward ~2,900
// and DSP 135 -> ~195 (89%). If cycles stop falling or resources approach
// 100%, that is the stopping point Dr. Grover described.
// BASELINE FOR THIS SWEEP (fixed weights, SLIDE_PAR 2/2):
//   5,641 cycles | 34,798 LUT (65%) | 26,552 FF (25%) | 135 DSP (61%)
//   | 47 BRAM (17%) | 7.213 ns
//
// EARLIER (float weights):
//   5,661 cycles | 113,986 LUT (214%) | 70,045 FF (66%) | 174 DSP (79%)
//   | 256 BRAM (91%) | 7.287 ns.  Stages: conv1+pool1 4,033 / conv2+pool2
//   5,660 (bound) / fc1 3,918 / fc2 3,191 / fc3 239.
// THIS FILE adds the fc3 area fix on top of that build, applied identically
// to all three designs, so the FC back-end stays a controlled constant.
// ============================================================================
//
// THE THREE BANKING VARIANTS (all generated from one template, so the ONLY
// difference between them is how the two convolution bands are partitioned):
//
//   variant          conv1 band          conv2 band            measured
//   2-D both      lb[6][32] cyc 8     lb[6][84] cyc 12    5,641 | 37,429 LUT | 139 DSP
//   flat 72       lb[192] complete    lb[504] cyclic 72   5,630 | 48,346 LUT | 188 DSP
//   mixed         lb[192] complete    lb[6][84] cyc 12    5,641 | 35,175 LUT | 139 DSP
//
// MIXED IS THE BEST of the three, and the reason is a size threshold, not a
// preference for either form:
//
//   conv1 reads 36 of 192 elements per cycle -> a 192-way mux is affordable,
//         and layering row separation on top costs more decode than it saves
//         (2-D measured 11,745 LUT / 13,549 FF versus 9,491 / 4,082 flat).
//   conv2 reads 72 of 504 elements per cycle -> a 504-way mux is not
//         affordable (35,557 LUT), and no single-level flat banking works:
//           factor 64/128/256  row bases spaced 84 wrap and collide
//           factor 72          conflict-free but not a power of two, so
//                              addr/72 and addr%72 become runtime divisions
//                              mapped onto DSPs (60 -> 109 DSP, LUT unchanged)
//           factor 512         array is 504 elements, identical to complete
//         The 2-D form expresses rows-then-columns directly and is
//         conflict-free at roughly 13,000 LUT.
//
// ---------------------------------------------------------------------------
// WHY FUSION: IT MOVES THE FIFO PORT FLOOR
// ---------------------------------------------------------------------------
// An hls::stream has one read and one write port, so a stage moving N values
// through a FIFO cannot finish in under N cycles. On the unfused designs this
// was measured directly, with Vitis reporting II equal to the accesses per
// iteration and naming the cause "Resource Limitation":
//
//     LOOP_C1_COL  II = 6   (6 FIFO writes)
//     LOOP_P1_COL  II = 12  (12 FIFO reads)
//     LOOP_P2_COL  II = 32  (32 FIFO reads)
//
// Unfused, conv1 writes 4,704 values and pool1 reads the same 4,704, so the
// network floor is 4,704 cycles regardless of how much logic is added.
// Fusion DELETES that stream: the fused stage reads 1,024 and writes 1,176.
//
//     network floor, unfused : 4,704 cycles
//     network floor, fused   : 1,176 cycles      4.0x lower
//
// This is the quantified form of the paper's data-movement claim. Note that
// it does NOT support a reduction in redundant computation: the same
// convolutions are still performed, they are simply never stored.
//
// ---------------------------------------------------------------------------
// SIX LINE BUFFERS, NOT FIVE
// ---------------------------------------------------------------------------
// A 5x5 window needs 5 rows, but the four slide positions reach one row
// further (rows 2ph .. 2ph+5). That one extra line buffer is the entire
// storage cost of fusion, and it replaces the pooling stage's two-row buffer,
// which is deleted.
//
// FORMULATION. Loop bounds count POOLED outputs. For each pooled output the
// kernel is slid to its four positions and the max of the four ReLU results
// is written straight out:
//
//     s = 0 -> (2ph  , 2pw  )      s = 1 -> (2ph  , 2pw+1)
//     s = 2 -> (2ph+1, 2pw  )      s = 3 -> (2ph+1, 2pw+1)
//
// All four windows read one shared 6x6 patch and use the SAME weights: 4x the
// arithmetic for 1x the weight bandwidth.
//
// WINDOW ADVANCE
//   horizontal : by ADDRESS, (2*pw + dw + kw). Never a data movement.
//   vertical   : by an UNROLLED two-row shift, 13 times (conv1) and 4 times
//                (conv2). dh and kh are compile-time constants, so the line
//                buffer number stays constant and no multiplexer forms.
//
// STREAM CONSUMPTION, verified exact
//   conv1+pool1  192 fill + 13*64  = 1024   (32*32*1)
//   conv2+pool2  504 fill +  4*168 = 1176   (14*14*6)
//
// EXACTNESS. Bit-identical to the unfused design: ReLU outputs are >= 0 so
// seeding the running max with 0 is exact, and fixed-point addition is
// associative so the ps[]/ms[] reassociation is exact.
//
// ---------------------------------------------------------------------------
// CONTROLLED COMPARISON - DO NOT DIVERGE THESE
// ---------------------------------------------------------------------------
// FC1_TILE 16, FC2_TILE 4 and the output-tiled fc3 below are byte-identical
// across all three designs, so the buffering/fusion architecture is the only
// variable in the comparison.
// ============================================================================
#include "lenet.h"
#include "parameters_fixed.h"

// ---------------------------------------------------------------------------
// DESIGN POINT
//   SLIDE_PAR = 1  sequential slides   (conv1  25 DSP, conv2  30 DSP)
//   SLIDE_PAR = 2  two windows at once (conv1  50 DSP, conv2  60 DSP)
//   SLIDE_PAR = 4  all four at once    (conv1 100 DSP, conv2 120 DSP)
//
// 2/2 is the balanced point at 139 of 220 DSP. For the first C-simulation
// after any structural edit set BOTH to 1: that is pure fusion with no added
// parallelism, and it must match the unfused predicted class exactly.
// ---------------------------------------------------------------------------
#define SLIDE_PAR_C1   2
#define SLIDE_PAR_C2   4
#define FC1_TILE       16      // must divide 120; matches the unfused designs
#define FC2_TILE       4
#define FC3_TILE       4      // must divide  84; matches the unfused designs

#define C1_GROUPS     (4 / SLIDE_PAR_C1)
#define C2_GROUPS     (4 / SLIDE_PAR_C2)

#define C1_ROWLEN   32
#define C2_ROWLEN   (14*6)               // 84
#define LB_DEPTH     6                   // line buffers per conv stage
#define C1_BAND     (LB_DEPTH*C1_ROWLEN) // 192, flat
// C2_BAND unused: conv2 is declared lb[LB_DEPTH][C2_ROWLEN]

// Line buffer n, column c.  <<< the indexing formula
#define C1_AT(n, c)  lb[(n)*C1_ROWLEN + (c)]   // flat 1-D
#define C2_AT(n, c)  lb[n][c]                  // 2-D, see BANKING NOTE

static inline data_t relu(data_t x){ return (x > 0) ? x : (data_t)0; }

// ===========================================================================
// STAGE A : conv1 (5x5, 1 -> 6) + ReLU + maxpool 2x2, fused
//           32x32x1  ->  14x14x6 emitted directly
// ===========================================================================
static void conv1_pool1_fused(hls::stream<data_t>& in, hls::stream<data_t>& out){
    // FLAT band. Per cycle this reads 6 rows x 6 consecutive columns = 36
    // values from only 192 elements, so the 192-way mux from complete
    // partitioning is affordable.
    //
    // MEASURED: the 2-D two-level scheme (dim=1 complete, dim=2 cyclic 8) is
    // WORSE here: 4,103 cycles / 11,745 LUT / 13,549 FF versus 3,977 / 9,491
    // / 4,082 flat. Layering row separation on a small array costs more decode
    // than the mux it removes, and factor 8 over-provisions a 32-wide row that
    // needs only 6 banks.
    data_t lb[C1_BAND];
    #pragma HLS ARRAY_PARTITION variable=lb type=complete
    #pragma HLS ARRAY_PARTITION variable=conv1_weights type=complete dim=0
    #pragma HLS ARRAY_PARTITION variable=conv1_bias    type=complete

    // Prime LB0..LB5 with input rows 0..5.
    LOOP_C1_FILL: for (int i = 0; i < C1_BAND; i++){
        #pragma HLS PIPELINE II=1
        lb[i] = in.read();
    }
    LOOP_C1_PH: for (int ph = 0; ph < 14; ph++){
        LOOP_C1_PW: for (int pw = 0; pw < 14; pw++){
            LOOP_C1_CO: for (int co = 0; co < 6; co++){

                data_t ms[C1_GROUPS];
                #pragma HLS ARRAY_PARTITION variable=ms type=complete

                LOOP_C1_SG: for (int sg = 0; sg < C1_GROUPS; sg++){
                    #pragma HLS PIPELINE II=1
                    data_t mg = 0;               // exact: candidates are >= 0

                    LOOP_C1_U: for (int u = 0; u < SLIDE_PAR_C1; u++){
                        #pragma HLS UNROLL
                        const int s  = sg * SLIDE_PAR_C1 + u;
                        const int dh = s >> 1;   // 0,0,1,1
                        const int dw = s &  1;   // 0,1,0,1

                        data_t sum = (data_t)conv1_bias[co];
                        // dh and kh are compile-time constants, so the line
                        // buffer number is constant and the read is a plain
                        // column select. No rotation, no multiplexer.
                        for (int kh = 0; kh < 5; kh++){
                            #pragma HLS UNROLL
                            for (int kw = 0; kw < 5; kw++){
                                #pragma HLS UNROLL
                        // C1_AT (Line buffer) = lb[(n)*C1_ROWLEN + (c)] 
                                sum += C1_AT(dh + kh, 2*pw + dw + kw)
                                     * (data_t)conv1_weights[co][0][kh][kw];
                            }
                        }
                        data_t v = relu(sum);
                        if (v > mg) mg = v;
                    }
                    ms[sg] = mg;
                }
                data_t m = ms[0];
                LOOP_C1_RED: for (int g = 1; g < C1_GROUPS; g++){
                    #pragma HLS UNROLL
                    if (ms[g] > m) m = ms[g];
                }
                out.write(m);
            }
        }

        // Rows 2ph and 2ph+1 are out of scope for ph+1. Shift the band up by
        // TWO rows and load rows 2ph+6 and 2ph+7 into LB4 and LB5. 13 shifts
        // per image, unrolled to about one cycle each.
        if (ph < 13){
            LOOP_C1_SHIFT: for (int i = 0; i < C1_BAND - 2*C1_ROWLEN; i++){
                #pragma HLS UNROLL
                lb[i] = lb[i + 2*C1_ROWLEN];               // LB0<-LB2 ... LB3<-LB5
            }
            LOOP_C1_LOAD: for (int i = 0; i < 2*C1_ROWLEN; i++){
                #pragma HLS PIPELINE II=1
                lb[C1_BAND - 2*C1_ROWLEN + i] = in.read(); // rows 2ph+6, 2ph+7
            }
        }
    }
    // reads: 192 + 13*2*32 = 1024
}

// ===========================================================================
// STAGE B : conv2 (5x5, 6 -> 16) + ReLU + maxpool 2x2, fused
//           14x14x6  ->  5x5x16 emitted directly
// ===========================================================================
static void conv2_pool2_fused(hls::stream<data_t>& in, hls::stream<data_t>& out){
    // BANKING NOTE - the one place this design is not a flat 1-D array.
    //
    // Per cycle conv2 reads 6 rows x 12 consecutive columns = 72 values.
    // Serving that needs a TWO-LEVEL scheme: separate the rows, then bank the
    // columns. A flat array accepts only ONE partition directive, and an
    // exhaustive search over the flat form found no usable factor:
    //
    //   factor 64/128/256  row bases spaced 84 wrap and collide
    //   factor 72          conflict-free, but not a power of two, so addr/72
    //                      and addr%72 cost DSPs (60 -> 109 DSP)
    //   factor 512         array is 504 elements, identical to complete
    //   complete           504-way mux per read (35,557 LUT, 92% of device)
    //
    // The 2-D form expresses the two levels directly and is conflict-free at
    // roughly 13,000 LUT. Row semantics are unchanged: lb[n] is line buffer n.
    data_t lb[LB_DEPTH][C2_ROWLEN];
    #pragma HLS ARRAY_PARTITION variable=lb dim=1 type=complete
    #pragma HLS ARRAY_PARTITION variable=lb dim=2 type=cyclic factor=12
    #pragma HLS ARRAY_PARTITION variable=conv2_weights type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=conv2_weights type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=conv2_bias    type=complete

    LOOP_C2_FILL_R: for (int r = 0; r < LB_DEPTH; r++)
        LOOP_C2_FILL_C: for (int i = 0; i < C2_ROWLEN; i++){
            #pragma HLS PIPELINE II=1
            lb[r][i] = in.read();
        }

    LOOP_C2_PH: for (int ph = 0; ph < 5; ph++){
        LOOP_C2_PW: for (int pw = 0; pw < 5; pw++){
            LOOP_C2_CO: for (int co = 0; co < 16; co++){

                data_t ms[C2_GROUPS];
                #pragma HLS ARRAY_PARTITION variable=ms type=complete

                LOOP_C2_SG: for (int sg = 0; sg < C2_GROUPS; sg++){

                    // ps[u][kw]: one partial sum per window per kernel column.
                    // Writing a distinct element each cycle removes the
                    // accumulator recurrence that otherwise forces II ~ 153.
                    data_t ps[SLIDE_PAR_C2][5];
                    #pragma HLS ARRAY_PARTITION variable=ps type=complete dim=0

                    LOOP_C2_KW: for (int kw = 0; kw < 5; kw++){
                        #pragma HLS PIPELINE II=1
                        LOOP_C2_U: for (int u = 0; u < SLIDE_PAR_C2; u++){
                            #pragma HLS UNROLL
                            const int s  = sg * SLIDE_PAR_C2 + u;
                            const int dh = s >> 1;
                            const int dw = s &  1;

                            data_t p = 0;
                            for (int ci = 0; ci < 6; ci++){
                                #pragma HLS UNROLL
                                for (int kh = 0; kh < 5; kh++){
                                    #pragma HLS UNROLL
                                    p += C2_AT(dh + kh, (2*pw + dw + kw)*6 + ci)
                                       * (data_t)conv2_weights[co][ci][kh][kw];
                                }
                            }
                            ps[u][kw] = p;
                        }
                    }

                    data_t mg = 0;
                    LOOP_C2_FOLD: for (int u = 0; u < SLIDE_PAR_C2; u++){
                        #pragma HLS UNROLL
                        data_t sum = (data_t)conv2_bias[co];
                        for (int kw = 0; kw < 5; kw++){
                            #pragma HLS UNROLL
                            sum += ps[u][kw];
                        }
                        data_t v = relu(sum);
                        if (v > mg) mg = v;
                    }
                    ms[sg] = mg;
                }

                data_t m = ms[0];
                LOOP_C2_RED: for (int g = 1; g < C2_GROUPS; g++){
                    #pragma HLS UNROLL
                    if (ms[g] > m) m = ms[g];
                }
                out.write(m);
            }
        }

        if (ph < 4){
            LOOP_C2_SHIFT_R: for (int r = 0; r < LB_DEPTH-2; r++)
                LOOP_C2_SHIFT_C: for (int i = 0; i < C2_ROWLEN; i++){
                    #pragma HLS UNROLL
                    lb[r][i] = lb[r+2][i];                 // LB0<-LB2 ... LB3<-LB5
                }
            LOOP_C2_LOAD_A: for (int i = 0; i < C2_ROWLEN; i++){
                #pragma HLS PIPELINE II=1
                lb[LB_DEPTH-2][i] = in.read();             // row 2ph+6 -> LB4
            }
            LOOP_C2_LOAD_B: for (int i = 0; i < C2_ROWLEN; i++){
                #pragma HLS PIPELINE II=1
                lb[LB_DEPTH-1][i] = in.read();             // row 2ph+7 -> LB5
            }
        }
    }
    // reads: 504 + 4*2*84 = 1176
}

// ===========================================================================
// STAGE C : FC1 (400 -> 120), buffered, output-tiled FC1_TILE wide
// Identical to the unfused designs.
// ===========================================================================
static void fc1_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    #pragma HLS ARRAY_PARTITION variable=fc1_weights dim=1 type=cyclic factor=FC1_TILE

    data_t xb[400];
    // Stage B emits (ph, pw, co); the trained flatten order is (co, ph, pw).
    LOOP_FC1_BUF: for (int i = 0; i < 400; i++){
        #pragma HLS PIPELINE II=1
        data_t v = in.read();
        int c = i % 16;
        int w = (i / 16) % 5;
        int h = (i / 80);
        xb[c*25 + h*5 + w] = v;
    }

    LOOP_FC1_OG: for (int og = 0; og < 120; og += FC1_TILE){
        data_t ag[FC1_TILE];
        #pragma HLS ARRAY_PARTITION variable=ag type=complete
        for (int k = 0; k < FC1_TILE; k++){
            #pragma HLS UNROLL
            ag[k] = (og + k < 120) ? (data_t)fc1_bias[og+k] : (data_t)0;
        }
        LOOP_FC1_J: for (int j = 0; j < 400; j++){
            #pragma HLS PIPELINE II=1
            data_t v = xb[j];
            for (int k = 0; k < FC1_TILE; k++){
                #pragma HLS UNROLL
                int oi = (og + k < 120) ? (og + k) : 119;   // clamp, in bounds
                data_t contrib = v * (data_t)fc1_weights[oi][j];
                if (og + k < 120) ag[k] += contrib;
            }
        }
        for (int k = 0; k < FC1_TILE; k++){
            #pragma HLS UNROLL
            if (og + k < 120) out.write(relu(ag[k]));   // only real outputs
        }
    }
}

// ===========================================================================
// STAGE D : FC2 (120 -> 84), buffered, output-tiled FC2_TILE wide
// ===========================================================================
static void fc2_stream(hls::stream<data_t>& in, hls::stream<data_t>& out){
    #pragma HLS ARRAY_PARTITION variable=fc2_weights dim=1 type=cyclic factor=FC2_TILE

    data_t xb[120];
    LOOP_FC2_BUF: for (int i = 0; i < 120; i++){
        #pragma HLS PIPELINE II=1
        xb[i] = in.read();
    }

    LOOP_FC2_OG: for (int og = 0; og < 84; og += FC2_TILE){
        data_t ag[FC2_TILE];
        #pragma HLS ARRAY_PARTITION variable=ag type=complete
        for (int k = 0; k < FC2_TILE; k++){
            #pragma HLS UNROLL
            ag[k] = (data_t)fc2_bias[og+k];
        }
        LOOP_FC2_J: for (int j = 0; j < 120; j++){
            #pragma HLS PIPELINE II=1
            data_t v = xb[j];
            for (int k = 0; k < FC2_TILE; k++){
                #pragma HLS UNROLL
                ag[k] += v * (data_t)fc2_weights[og+k][j];
            }
        }
        for (int k = 0; k < FC2_TILE; k++) out.write(relu(ag[k]));
    }
}

// ===========================================================================
// STAGE E : FC3 (84 -> 43) + argmax + AXI packing
//
// Arithmetic is byte-identical to the unfused designs' fc3. The only addition
// is the AXI word, built here rather than in lenet_once so the DATAFLOW region
// contains nothing but process calls.
//
// is_last carries TLAST in from the batch loop. Asserting TLAST on every
// packet ends the DMA transfer after the first image and hangs the S2MM
// channel on hardware; the testbench checks for exactly this.
// ===========================================================================
static void fc3_stream(hls::stream<data_t>& in, hls::stream<axis_out_t>& out,
                       bool is_last){
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
    {   // VERIFICATION: all 43 accumulators against the reference scores.
        // Compiled out of synthesis; costs no hardware. Verified 43/43 exact
        // (worst delta 0.0000, class 29) on this design at SLIDE_PAR 1/1.
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
        printf("fc3 verify: %d/43 exact  worst %.6f  class %d (golden %d)  %s\n",
               matches, worst, cmi, gmi, cmi == gmi ? "OK" : "*** MISMATCH ***");
    }
#endif
    int mi = 0; data_t mv = acc[0];
    for (int i = 1; i < 43; i++){ if (acc[i] > mv){ mv = acc[i]; mi = i; } }

    axis_out_t o;
    o.data = mi;
    o.last = is_last ? 1 : 0;
    // If axis_out_t has no keep/strb side channels, delete the next two lines;
    // copy the exact field names from lenet.h.
    o.keep = -1;
    o.strb = -1;
    out.write(o);
}

// ===========================================================================
// DATAFLOW REGION : five stages, four FIFOs
// (the unfused design needs seven stages and six FIFOs)
// Every statement here must be a process call, or HLS silently drops the
// dataflow optimisation.
// ===========================================================================
static void lenet_once(hls::stream<data_t>& input,
                       hls::stream<axis_out_t>& output,
                       bool is_last){
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    hls::stream<data_t> s1, s2, s3, s4;
    #pragma HLS STREAM variable=s1 depth=64
    #pragma HLS STREAM variable=s2 depth=64
    #pragma HLS STREAM variable=s3 depth=64
    #pragma HLS STREAM variable=s4 depth=64

    conv1_pool1_fused(input, s1);        //  32x32x1 -> 14x14x6
    conv2_pool2_fused(s1,    s2);        //  14x14x6 ->  5x5x16
    fc1_stream       (s2,    s3);        //     400  ->  120
    fc2_stream       (s3,    s4);        //     120  ->   84
    fc3_stream       (s4,    output, is_last);
}

// ===========================================================================
// TOP LEVEL
// ===========================================================================
void lenet_predict(hls::stream<data_t>& input,
                   hls::stream<axis_out_t>& output,
                   int batches){
    #pragma HLS INTERFACE axis      port=input
    #pragma HLS INTERFACE axis      port=output
    #pragma HLS INTERFACE s_axilite port=batches bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control

    LOOP_BATCH: for (int b = 0; b < batches; b++){
        lenet_once(input, output, b == batches - 1);
    }
}
