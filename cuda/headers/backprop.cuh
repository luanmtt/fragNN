#pragma once
#include <cuda_runtime.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// backward pass kernels — backprop.cu


__global__ void matmul_backp_X( float* W,
                                float* dl_dy,   
                                float* dl_dx,
                                int batch,
                                int in_dim,
                                int out_dim);

__global__ void matmul_backp_W( float* X,
                                float* dl_dy,
                                float* dl_dw,
                                int batch,
                                int in_dim, 
                                int out_dim);

__global__ void matmul_backp_b( float* dl_dy,   
                                float* dl_db,
                                int batch,
                                int out_dim);

__global__ void activation_backp(   float* X,
                                    float* dl_dy,
                                    float* dl_dx,
                                    int n,
                                    int activation_type);

__global__ void softmax_backp(  float* probabilities,
                                float* dl_dy,
                                float* dl_dx,
                                int batch,
                                int n_classes);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
