#include "kernels.cuh"
#include "headers/activations.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__ void matmul_backp_X( float* X,
                                float* dl_dy,   
                                float* dl_dx,
                                int batch,
                                int in_dim,
                                int out_dim){
    
   
     


}


__global__ void matmul_backp_W( float* X,
                                float* dl_dy,
                                float* dl_dw,
                                int batch,
                                int in_dim, 
                                int out_dim){


    /*
        // how much did each weight contribute to the error?
        // dL/dW = Xᵀ @ dL_dout

        one thread per (in_dim, out_dim) pair — one element of dL_dW
        row = i / out_dim    ← which input neuron
        col = i % out_dim    ← which output neuron

        sum = 0
        for each sample in batch:
            sum += X[sample * in_dim + row] * dL_dout[sample * out_dim + col]
            //     Xᵀ[row, sample]            dL_dout[sample, col]

        dL_dW[row * out_dim + col] = sum

    */
    
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= batch * in_dim)
        return;
    
    const int row = i / out_dim;
    const int col = i % out_dim;

    float sum = 0.0f;

    for(int k = 0; k < in_dim; k++){
        sum += X[k * in_dim + row] * dl_dy[k * out_dim + col];

    }
        
    dl_dw[row * out_dim + col] = sum;

}



__global__ void matmul_backp_b( float* dl_dy,   
                                float* dl_db,
                                int batch,
                                int out_dim){

    /*
        // how much did each bias contribute to the error?
        // dL/db = sum(dL_dout) over batch dimension

        one thread per output neuron — one element of dL_db
          i = thread index, bounded by out_dim

          sum = 0
          for each sample in batch:
            sum += dL_dout[sample * out_dim + i]
            // bias is shared across all samples
            // so we sum gradients from all of them

          dL_db[i] = sum
    */

    /*
        lógica para backp_bias: 

        ◦   a presença de vieses para cada perceptron causa
            um deslocamento constante no resultado final. talvez
            seja de lá que vem os erros da IA! logo, precisamos
            corriji-los constantemente. pegamos a soma local porque 
            o vies local é compartilhado no perceptron.

            Vamos escrever em dl_dy essa relação.
        
            
            • Indexação: presente no for-loop de sum.
                
            ex: dl_dy = [0.2, 0.4, 0.5, 0.6, 0.1, 0.2, 1.0]
                batch = 2
                out_dim = 3

            thread 0 → i=0            
                0:  sum += dl_dy[0 * 3 + 0] , += 0.2
                1:  sum += dl_dy[1 * 3 + 0] , += 0.6
                2:  sum += dl_dy[2 * 3 + 0] , += 1.0

            thread 1 → i=1 
                0:  sum += dl_dy[0 * 3 + 1] , += 
                1:  sum += dl_dy[1 * 3 + 1] , +=
                2:  sum += dl_dy[2 * 3 + 1] , +=

    */
        
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= batch * out_dim)
        return;

    float sum = 0.0f;

    for(int k = 0; k < batch; k++){
        
        sum += dl_dy[i * out_dim + k];
    }
       
    dl_db[i] = sum;

}


__global__ void activation_backp(   float* X,
                                    float* dl_dy,
                                    float* dl_dx,
                                    int n,
                                    int activation_type){

    /*
        // chain rule through the activation function
        // dL/dX = dL_dout * activation'(X)

        one thread per element, bounded by n_elements
          switch on ActivationType:
            LEAKY_RELU → dL_dX[i] = dL_dout[i] * leaky_relu_grad(X[i])
            SIGMOID    → dL_dX[i] = dL_dout[i] * sigmoid_grad(X[i])
            TANH       → dL_dX[i] = dL_dout[i] * tanh_grad(X[i])
    */

    const int i = blockIdx.x * blockIdx.x + threadIdx.x;
    if(i >+ n)
        return;

    switch(activation_type){

        case 0: { // sigmoid
            dl_dx[i] = dl_dy[i] * sigmoid_backp(X[i]);
        } 
        
        case 1: { // relu
            dl_dx[i] = dl_dy[i] * relu_backp(X[i], 0.01f);
        }

        case 2: { // leaky_relu
            dl_dx[i] = dl_dy[i] * leaky_relu_backp(X[i], 0.01f);
        }
        
        case 3: { //
            dl_dx[i] = dl_dy[i] * tanh_backp(X[i]);
        }
        
        default: {
            printf("[ERRO]: Opção '%d' de ActivationType equivocada. Retornando...", activation_type);
            return;
        }
    }

}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
