// ============================================================================
// tb_lenet.cpp — testbench for the four-window-slide fused LeNet-5
//
// Matches the 3-argument top level declared in lenet.h:
//     void lenet_predict(hls::stream<data_t>&,
//                        hls::stream<axis_out_t>&,
//                        int batches);
//
// PURPOSE
// -------
// This checks BIT-EXACTNESS against the unfused two-array design, not
// classification accuracy. Run the two-array project on the same image,
// note the class it prints, put that number in GOLDEN_CLASS below, and
// this testbench will confirm the fused design agrees.
//
// Set GOLDEN_CLASS to -1 to skip the comparison and just print the result.
//
// The image is streamed ROW-MAJOR, 32 rows of 32 pixels, because
// conv1_pool1_fused reads rows sequentially into its circular buffer.
// ============================================================================
#include "lenet.h"
#include "image_data.h"
#include <cstdio>

#define IMG_H   32
#define IMG_W   32
#define IMG_N   (IMG_H * IMG_W)
#define NBATCH  1

// Expected class from the two-array reference run. -1 disables the check.
#define GOLDEN_CLASS  (-1)

int main(){

    hls::stream<data_t>     input("input");
    hls::stream<axis_out_t> output("output");

    // ---- feed NBATCH images, row-major ------------------------------------
    for (int b = 0; b < NBATCH; b++){
        for (int i = 0; i < IMG_N; i++){
            input.write((data_t)test_image[0][i / IMG_W][i % IMG_W]);
        }
    }

    printf("streamed %d pixels for %d image(s)\n", IMG_N * NBATCH, NBATCH);

    // ---- run the accelerator ----------------------------------------------
    lenet_predict(input, output, NBATCH);

    // ---- drain and report --------------------------------------------------
    int fails = 0;

    for (int b = 0; b < NBATCH; b++){
        if (output.empty()){
            printf("ERROR: no output word for image %d\n", b);
            fails++;
            continue;
        }
        axis_out_t o = output.read();
        int predicted_class = (int)o.data;      // [ADJUST] AXI field name

        if (GOLDEN_CLASS < 0){
            printf("image %d : predicted class %d\n", b, predicted_class);
        } else {
            bool ok = (predicted_class == GOLDEN_CLASS);
            printf("image %d : predicted %2d, golden %2d  %s\n",
                   b, predicted_class, GOLDEN_CLASS, ok ? "MATCH" : "MISMATCH");
            if (!ok) fails++;
        }
    }

    // ---- leftover data is a bug, not a warning -----------------------------
    if (!input.empty()){
        int left = 0;
        while (!input.empty()){ input.read(); left++; }
        printf("ERROR: %d input samples unconsumed\n", left);
        fails++;
    }
    if (!output.empty()){
        int left = 0;
        while (!output.empty()){ output.read(); left++; }
        printf("ERROR: %d extra output words\n", left);
        fails++;
    }

    if (fails == 0){
        printf("\nPASS\n");
        return 0;
    } else {
        printf("\nFAIL (%d)\n", fails);
        return 1;
    }
}
