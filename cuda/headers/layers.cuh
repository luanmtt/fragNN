#pragma once
#include <cuda_runtime.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__ void matmul( float* X, 
                        float* Y,
                        float* W,
                        float *b,
                        int batch,
                        int in_dim, 
                        int out_dim);

__global__ void apply_activation(float* X, int n, int activation_type);

__global__ void softmax(float* X, int batch, int n_classes);


// ─────────────────────────────────────────────────────────────────────────────────────────────────


__global__ void matmul_backp_X( float* X,
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
