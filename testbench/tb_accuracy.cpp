// ============================================================================
// tb_accuracy.cpp - GTSRB test-set accuracy of the ap_fixed<16,8> design
//
// Replaces tb_lenet.cpp as the project testbench. Runs in C SIMULATION only:
// no board, no bitstream, no synthesis. ap_fixed is a C++ class, so the
// rounding here is bit-identical to what the hardware does - the number this
// prints is the accuracy of the design you are shipping.
//
// Reads gtsrb_test.bin, written either by tb_make_selftest.cpp (small, for
// checking the harness) or by export_testset.py (the real test set).
//
// SET THE PATH BELOW. Vitis HLS runs C simulation from
//   <project>/<solution>/csim/build/
// so a bare filename will not be found. Use the full path.
// ============================================================================
#include "lenet.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define IMG_H   32
#define IMG_W   32
#define IMG_N   (IMG_H * IMG_W)

// [EDIT THIS] absolute path, forward slashes work on Windows too
static const char *TEST_FILE =
    "C:/Users/jeong/TEST_FINAL_Research/FINAL_fusion/gtsrb_subset.bin";

int main() {

    // ---- load the test set -------------------------------------------------
    FILE *f = std::fopen(TEST_FILE, "rb");
    if (!f) {
        std::printf("ERROR: cannot open %s\n", TEST_FILE);
        std::printf("Set TEST_FILE to the file's absolute path.\n");
        return 1;
    }

    int count = 0, h = 0, w = 0;
    if (std::fread(&count, sizeof(int), 1, f) != 1 ||
        std::fread(&h,     sizeof(int), 1, f) != 1 ||
        std::fread(&w,     sizeof(int), 1, f) != 1) {
        std::printf("ERROR: short read on header\n");
        std::fclose(f);
        return 1;
    }
    if (h != IMG_H || w != IMG_W) {
        std::printf("ERROR: file holds %dx%d images, expected %dx%d\n",
                    h, w, IMG_H, IMG_W);
        std::fclose(f);
        return 1;
    }

    std::vector<float> pixels((size_t)count * IMG_N);
    std::vector<int>   labels((size_t)count);

    if (std::fread(pixels.data(), sizeof(float), pixels.size(), f)
            != pixels.size() ||
        std::fread(labels.data(), sizeof(int), labels.size(), f)
            != labels.size()) {
        std::printf("ERROR: short read on data\n");
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    // Sanity check: image_data.h is normalised to [-1, 1]. Anything else
    // means the preprocessing does not match what the network was trained on.
    float lo = pixels[0], hi = pixels[0];
    for (size_t i = 1; i < pixels.size(); i++) {
        if (pixels[i] < lo) lo = pixels[i];
        if (pixels[i] > hi) hi = pixels[i];
    }
    std::printf("loaded %d images, pixel range %.4f to %.4f\n", count, lo, hi);
    if (lo < -1.05f || hi > 1.05f) {
        std::printf("\nSTOP: range is outside [-1, 1]; preprocessing does not\n");
        std::printf("      match image_data.h. The result would be meaningless.\n");
        return 1;
    }
    std::printf("\n");

    // ---- run every image ---------------------------------------------------
    int correct = 0;

    for (int n = 0; n < count; n++) {

        hls::stream<data_t>     input("input");
        hls::stream<axis_out_t> output("output");

        // Row-major, same order as tb_lenet.cpp. The (data_t) cast is where
        // the 16-bit quantisation happens - that is the measurement.
        const float *img = &pixels[(size_t)n * IMG_N];
        for (int i = 0; i < IMG_N; i++) {
            input.write((data_t)img[i]);
        }

        lenet_predict(input, output, 1);

        if (output.empty()) {
            std::printf("ERROR: no output for image %d\n", n);
            return 1;
        }
        int predicted = (int)output.read().data;

        if (predicted == labels[n]) correct++;

        // Leftover data means a stage mis-consumed - a real bug, so stop
        // rather than report a wrong number.
        if (!input.empty()) {
            std::printf("ERROR: image %d left input samples unconsumed\n", n);
            return 1;
        }
        if (!output.empty()) {
            std::printf("ERROR: image %d produced extra output words\n", n);
            return 1;
        }

        if ((n + 1) % 1000 == 0) {
            std::printf("  %5d / %d   running %.2f%%\n",
                        n + 1, count, 100.0 * correct / (n + 1));
        }
    }

    // ---- report ------------------------------------------------------------
    std::printf("\n");
    std::printf("images    : %d\n", count);
    std::printf("correct   : %d\n", correct);
    std::printf("ACCURACY  : %.2f%%\n", 100.0 * correct / count);
    std::printf("\n");
    std::printf("Pair this with the floating-point number from\n");
    std::printf("export_testset.py; the difference is what Reviewer 3 asked for.\n");

    return 0;
}