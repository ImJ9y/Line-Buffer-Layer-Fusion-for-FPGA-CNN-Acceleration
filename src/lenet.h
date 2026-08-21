#ifndef LENET_H
#define LENET_H
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>   // ap_axis for the TLAST-carrying output

#include <cmath>
#include <iostream>

// Fixed-point type used throughout the design
typedef ap_fixed<16,8> data_t;

// Output AXI-Stream packet: 32-bit prediction + side channels (TLAST etc.)
typedef ap_axis<32, 0, 0, 0> axis_out_t;

// Input Dimensions
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 1

// Layer 1: Conv 5x5, 6 Filters
#define C1_H 28
#define C1_W 28
#define C1_CH 6
#define C1_K 5

// Layer 2: MaxPool 2x2
#define S2_H 14
#define S2_W 14
#define S2_CH 6
#define S2_K 2

// Layer 3: Conv 5x5, 16 Filters
#define C3_H 10
#define C3_W 10
#define C3_CH 16
#define C3_K 5

// Layer 4: MaxPool 2x2
#define S4_H 5
#define S4_W 5
#define S4_CH 16
#define S4_K 2

// Fully Connected Layers
#define FC1_UNITS 120
#define FC2_UNITS 84
#define OUTPUT_CLASSES 43

//  Top Function (BATCHED: input stream, ap_axis output stream, batch count)
void lenet_predict(hls::stream<data_t> &input, hls::stream<axis_out_t> &output, int batches);

#endif