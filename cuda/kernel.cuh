#pragma once

#include <cuda_runtime.h>
#include <cstdio>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// utils:


typedef enum {

    ACT_SIGMOID     = 0,
    ACT_RELU        = 1,
    ACT_LEAKY_RELU  = 2,
    ACT_TANH        = 3

} ActivationType;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// sub-headers:


#include "headers/activations.cuh"
#include "headers/loss.cuh"
#include "headers/fwd.cuh"
#include "headers/backprop.cuh"
#include "headers/utils.cuh"
#include "headers/init.cuh"
#include "headers/metrics.cuh"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
