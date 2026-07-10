#include "kernel.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__global__ void HE(float* W, int n, int fan_in){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;
    
    curandState state;
    curand_init(50, i, 0, &state);
    float rand = curand_normal(&state);
    
    W[i] = rand * sqrtf(2.0f / (float) fan_in);

}

__global__ void init_b(float* b, int n){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;

    b[i] = 0.0f;
}


__global__ void zero_grad(float* dW, int n){

    /*
        // zero out gradient buffer before next step
        // one thread per element — total threads = n

        dW[i] = 0.0f
    */

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;

    dW[i] = 0.0f;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
