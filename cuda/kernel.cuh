#pragma once

#include <cuda_runtime.h>


/*
    
    Dicionary:

        X: inputs to any functions.
        Y: outputs to a function.
        W: weights of the parameters.
        b: bias of a defined layer.
        
        batch: batch_size processed at once.
        in: input num to a layer.
        out: output num from a layer.
        n_elements: batch * out
        n_classes: number of output classes (3: gender; 84: accords)
    
        logits: output of a layer before softmax.
        probabilities: outputs after softmax.

        mask: dropout mas: 0 if dropped, 1/keep_prob if kept.
        keep_prob: probability of a neuron staying alive.
        seed: random seed for keep_prob.

        dl_dy: gradient of the cost in respect to layer's output.
        dl_dx: gradient of the cost in respect to layer's input.
        dl_dw: gradient with respect to W.
        dl_db: gradient with respect to b.
*/



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// utils:


typedef enum {

    ACT_SIGMOID     = 0,
    ACT_RELU        = 1,
    ACT_LEAKY_RELU  = 2,
    ACT_TANH        = 3

} ActivationType;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// activation functions: estão em actiavtions.cu


__device__ float sigmoid(float x);
__device__ float sigmoid_backp(float x_activated);

__device__ float leaky_relu(float x, float alpha);
__device__ float leaky_relu_backp(float x_activated, float alpha);

__device__ float relu(float x, float alpha);
__device__ float relu_backp(float x_activated);

__device__ float tanh_(float x);
__device__ float tanh_backp(float x_activated);



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// loss functions: estão em loss.cu


__device__ float mse(float* pred, float* target, int n);

__device__ float cross_entropy(float* probabilities, int label, int n_classes);

__device__ float focal( float* probabilities, 
                        int label, 
                        int n_classes, 
                        float gamma);

__device__ float huber(float pred, float target, float delta);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// matmuls & layers: estão em fwd.cu


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
// backprop: estão em layers.cu

 
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
// optimizers: estão em utils.cu


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
