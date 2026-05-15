#include "kernel.cuh"
#include "headers/activations.cuh"

#include <cmath>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// fwdp 


__device__ float sigmoid(float x){

    return 1.0f/(1.0f +__powf(M_E, -x));
}


__device__ float leaky_relu(float x, float alpha = 0.01f){

    if(x < 0)
        return alpha * x;
    else
        return x;
}


__device__ float relu(float x){

    if(x < 0)
        return 0.0f;
    else
        return x;
}



__device__ float tanh_(float x){

    return (__powf(M_E, x) - __powf(M_E, -x))/(__powf(M_E, x) + __powf(M_E, -x));

}


// ─────────────────────────────────────────────────────────────────────────────────────────────────
// backp equivallents


__device__ float sigmoid_backp(float x_activated){
    
    float z = sigmoid(x_activated);

    return z * (1 - z);

}


__device__ float leaky_relu_backp(float x_activated, float alpha){
    
    float z = leaky_relu(x_activated);
    
    if(z > 0)
        return 1;
    else
        return alpha;

}


__device__  float relu_backp(float x_activated){

    float z = relu(x_activated);

    if(z > 0)
        return 1;
    else
        return 0;

}


__device__ float tanh_backp(float x_activated){

    float z = tanh(x_activated);
     
    return 1 - (z * z);

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
