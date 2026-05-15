#include "kernels.cuh"
#include "headers/utils.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__ void adam(   float* W,
                        float* dl_dw,
                        float* m,
                        float* v,
                        float lr,
                        float beta_1,
                        float beta_2,
                        float epsilon,
                        int timestamp,  
                        int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per weight
    if (i >= n) return;

    float m_t, v_t, m_t2, v_t2;
    m_t = v_t = m_t2 = v_t2 = 0;
    
    m_t = beta_1 * m[i] + (1 - beta_1) * dl_dw[i];
    v_t = beta_2 * v[i] + (1 - beta_2) * dl_dw[i] * dl_dw[i];

    m[i] = m_t;
    v[i] = v_t;

    m_t2 = m_t / (1 - __powf(beta_1, timestamp)); 
    v_t2 = v_t / (1 - __powf(beta_2, timestamp)); 
    
    W[i] = W[i] - lr * m_t2 / (sqrtf(v_t2) + epsilon); 
    
}


__global__ void dropout_forward(float* activations, 
                                float* mask,
                                float keep_prob, 
                                int n,
                                unsigned long long seed){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    curandState state;
    curand_init(seed, i, 0, &state);
    float rand_num = curand_uniform(&state);

    float* output;

    if(rand_num > keep_prob)
        mask[i] = 0.0f;
    else
        mask[i] = 1.0f / keep_prob;

    output[i] = activations[i] * mask[i];
     

}


__global__ void dropout_backprop(   float* dl_dy,
                                    float* dl_dx,
                                    float* mask,
                                    int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

        dl_dx[i] = dl_dy[i] * mask[i];

}





// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
