#pragma once
#include <cuda_runtime.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// forward pass kernels — fwd.cu


__global__ void matmul( float* X, 
                        float* Y,
                        float* W,
                        float *b,
                        int batch,
                        int in_dim, 
                        int out_dim);

__global__ void apply_activation(float* X, int n, int activation_type);

__global__ void softmax(float* X, int batch, int n_classes);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
