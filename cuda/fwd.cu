#include "kernels.cuh"
#include "headers/activations.cuh"

#include <float.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__global__ void matmul( float* X,
                        float* Y,
                        float* W,
                        float* b,
                        int batch,
                        int in_dim,
                        int out_dim){
  
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= batch * out_dim) 
        return;
        
    /*  
        indexação da seguinte maneira:
            
            X (2,3):          W (3,2):          Y (2,2):
            [1, 2, 3]         [1, 4]            [?, ?]
            [4, 5, 6]         [2, 5]            [?, ?]
                              [3, 6]
                
            → tem-se:   batch = 2;
                        in_dim = 3;
                        out_dim = 2;
                        
                        i = 4.
                
            ◦ k vai iterar os dígitos em linha, e i, que simboliza todas
            as threads do kernel, os aloca nos locais corretos em coluna.
            
            thread 0 → i=0 → row = 0/2 = 0,  col = 0%2 = 0
              k=0: X[0*3+0] * W[0*2+0] = X[0] * W[0] = 1*1 = 1
              k=1: X[0*3+1] * W[1*2+0] = X[1] * W[2] = 2*2 = 4
              k=2: X[0*3+2] * W[2*2+0] = X[2] * W[4] = 3*3 = 9
              
              ◦ Y[0] = 1+4+9 + b[0] = 14

            thread 1 → i=1 → row = 1/2 = 0,  col = 1%2 = 1
              k=0: X[0*3+0] * W[0*2+1] = X[0] * W[1] = 1*4 = 4
              k=1: X[0*3+1] * W[1*2+1] = X[1] * W[3] = 2*5 = 10
              k=2: X[0*3+2] * W[2*2+1] = X[2] * W[5] = 3*6 = 18
              
              ◦ Y[1] = 4+10+18 + b[1] = 32

            thread 2 → i=2 → row = 2/2 = 1,  col = 2%2 = 0
              k=0: X[1*3+0] * W[0*2+0] = X[3] * W[0] = 4*1 = 4
              k=1: X[1*3+1] * W[1*2+0] = X[4] * W[2] = 5*2 = 10
              k=2: X[1*3+2] * W[2*2+0] = X[5] * W[4] = 6*3 = 18
              
              ◦ Y[2] = 4+10+18 + b[0] = 32

            thread 3 → i=3 → row = 3/2 = 1,  col = 3%2 = 1
              k=0: X[1*3+0] * W[0*2+1] = X[3] * W[1] = 4*4 = 16
              k=1: X[1*3+1] * W[1*2+1] = X[4] * W[3] = 5*5 = 25
              k=2: X[1*3+2] * W[2*2+1] = X[5] * W[5] = 6*6 = 36
              
              ◦ Y[3] = 16+25+36 + b[1] = 77

    */
    

    const int row = i / out_dim;
    const int col = i % out_dim;
    
    float sum = 0.0f;

    for(int k = 0; k < in_dim; k++){
       
        sum += X[row * in_dim + k] * W[k * out_dim + col]; 
    }
    
    Y[i] = sum + b[col];

}


__global__ void apply_activation(float* X, int n, int activation_type){
    

    const int i = blockIdx.x + blockDim.x + threadIdx.x;    
    if(i >= n)
        return;
    
    switch(activation_type){

        case 0: { // sigmoid
            X[i] = sigmoid(X[i]);
        } 
        
        case 1: { // relu
            X[i] = relu(X[i], 0.01f);
        }

        case 2: { // leaky_relu
            X[i] = leaky_relu(X[i], 0.01f);
        }
        
        case 3: { //
            X[i] = tanh_(X[i]);
        }

        default: {
            printf("[ERRO]: Opção '%d' de ActivationType equivocada. Retornando...", activation_type);
            return;
        }
    }
}



__global__ void softmax(float* X, int batch, int n_classes){


    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= batch)
        return;
        
    /*
        em softmax, indexamos em respeito à linha que estamos.
        logo, se temos um array maçiço, precisamos posicionar
        a "row" no local correto:
 
        X = [ 0, 1, 2, 3, 4, 5 ]

        batch = 2
        n_classes = 3
        
        thread 0 → i=0 → row = [0] + 0 * 3 = [0]
                                             [0, 1, 2] no loop
                

        thread 1 → i=1 → row = [0] + 1 * 3 = [3]
                                             [3, 4, 5] no loop

        
        O softmax faz números, a primeira vista aletórios, serem relacionados como
        probabilidades. Primeiro, se captura o valor máximo. Depois, se exponenciam
        todos os valores: tiramos os negativos e introduzimos uma diferença de exponencial
        entre os valores: "e^3 is not just slightly bigger than e^1, it's 7x bigger"
        
        A subtração de max é feito para não termos um valor explosivo para o máximo ou valores
        próximos do máximo. É capturada a soma para uso futuro.

        Depois, normalizamos os valores usando row / sum. Isso gera valores menores que 1, em que a soma
        deles será 1 → Função de Probabilidade!

        A interpretação final é: 
            
            ◦ raw:  [1.0,  2.0,  3.0] 
                ↓
            ◦ probs: [0.09, 0.24, 0.66]: ∑ = 1
                ↓
            ◦ After softmax: "this fragrance is 66% likely class 2, 24% class 1, 9% class 0".

    */


    float* row = X + i * n_classes;
    float local_maxima = row[0];

    for(int k = 0; k < n_classes; k++)
        local_maxima = fmaxf(local_maxima, row[k]);    

    float sum = 0.0f;

    for(int k = 0; k < n_classes; k++){

        row[k] = __expf(row[k] - local_maxima);
        sum += row[k]; 

    }

    for(int k = 0; k < n_classes; k++)
        row[k] /= sum;

}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
