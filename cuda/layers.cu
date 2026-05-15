#include "kernels.cuh"
#include "headers/layers.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__global__ void matmul( float* X,
                        float* Y,
                        float* W,
                        float* b,
                        int batch,
                        int in_dim,
                        int out_dim){

     /*
        
        for each output element Y[i,j]:
            one thread handles one (i,j) pair
            Y[i,j] = sum over k of X[i,k] * W[k,j] + b[j]

     */
        
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int n = batch * out_dim; 
    if(i >= n) return;

    int row = i / out_dim;
    int col = i % out_dim;
    
    printf("TARGET: %d, %d\n", row, col);

    float sum = 0.0f; 
    for(int k = 0; k < in_dim; k++){

        sum += X[row * in_dim + k] * W[k * out_dim + col];
    }
        


}

__global__ void apply_activation(float* X, int n, int activation_type){







}



__global__ void softmax(float* X, int batch, int n_classes);



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
