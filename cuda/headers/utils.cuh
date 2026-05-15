#pragma once
#include <cuda_runtime.h>


#include <curand_kernel.h>
#include <cmath>
#include <cstdlib>

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__ void adam(   float* X,
                        float* dl_dw,
                        float* m,
                        float* v,
                        float lr,
                        float beta_1,
                        float beta_2,
                        float epsilon,
                        int timestamp,  
                        int n);


__global__ void dropout_forward(float* activations,
                                float* mask,
                                float keep_prob,
                                int n,
                                unsigned long long seed);

__global__ void dropout_backprop(   float* dl_dy,
                                    float* dl_dx,
                                    float* mask,
                                    int n);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
