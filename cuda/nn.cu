#include "kernel.cuh"
#include "../src/headers/dataloader.hpp"

#include <ctime>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


struct Layer {

    int in_dim; 
    int out_dim;
    int activation_type; // from ActivationType enum
    
    float* W;   // in * out
    float* b;   // out

    float* dW;  // in * out
    float* db;  // out

    float* m_W; // in * out
    float* v_W; // in * out

    float* m_b;  // out
    float* v_b;  // out
   
    float* mask; // out

};


struct Network {
    
    Layer* layers;
    
    // hyperparams
    int n_layers;
    float lr;

    float beta_1;
    float beta_2;
    float epsilon;          
    float keep_prob;        // dropout
    int timestamp;          // adam

    int num_features;

};


class NeuralNetwork {
    
    private:

        int NUM_BLOCKS;
        int BLOCK_SIZE = 256;
        bool training;

    public:
            
        
        void _set_NUM_BLOCKS_(int num_W){
            NUM_BLOCKS = (num_W + BLOCK_SIZE-1) / BLOCK_SIZE;
        }


        Network create_network( int* layers_dims,
                                int n_layers,
                                int middle_act,
                                int out_act,
                                float lr,
                                float keep_prob){

            Network net;
            net.keep_prob = keep_prob;
            net.lr = lr;
            net.n_layers = n_layers;
            net.timestamp = 0;

            // MAGIC NUMBERS
            net.beta_1 = 0.9f;
            net.beta_2 = 0.999f;
            net.epsilon = 1e-8f;

            int in_dim = 0;
            int out_dim = 0;
            int num_W = 0;
            int num_b = 0;
            
            net.layers = new Layer[n_layers];

            for(int i = 0; i < n_layers-1; i++){
                
                in_dim  = layers_dims[i];
                out_dim = layers_dims[i+1];
                num_W   = in_dim * out_dim;
                num_b   = out_dim;
                
                Layer& L = net.layers[i];
                L.in_dim = in_dim;
                L.out_dim = out_dim;
                
                if(i != n_layers-1){
                    L.activation_type = middle_act;

                } else {
                    L.activation_type = out_act;
                }   

                cudaMalloc(&L.W, num_W * sizeof(float));
                cudaMalloc(&L.b, num_b * sizeof(float));

                cudaMalloc(&L.dW, num_W * sizeof(float));
                cudaMalloc(&L.db, num_b * sizeof(float));
    
                cudaMalloc(&L.m_W,  num_W * sizeof(float));
                cudaMalloc(&L.v_W,  num_W * sizeof(float));

                cudaMalloc(&L.m_b,  num_b * sizeof(float));
                cudaMalloc(&L.v_b,  num_b * sizeof(float));

                cudaMalloc(&L.mask, num_b * sizeof(float));
                
                if(i == 0) _set_NUM_BLOCKS_(num_W);

                HE<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.W, num_W, in_dim);
                
                init_b<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.b, num_b);
 
                // d's
                zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.dW, num_W);

                zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.db, num_b);

                // W
                zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.m_W, num_W);

                zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.v_W, num_W);
            
                // b
                zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.m_b, num_b);

                zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.v_b, num_b);

                zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (L.mask, num_b);

            }

        }
        

        void fwd_pass(  Network network,
                        Layer& current, 
                        float* X,
                        float* Y,
                        int batch){
            

            matmul<<<NUM_BLOCKS, BLOCK_SIZE>>>  
            (X, Y, 
            current.W, current.b,
            batch, current.in_dim,
            current.activation_type);
 
            apply_activation<<<NUM_BLOCKS, BLOCK_SIZE>>>    
            (Y, batch * current.out_dim,
            current.activation_type);

            if(training){

                dropout_forward<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (Y, current.mask,  
                network.keep_prob,  batch * current.in_dim, 
                time(NULL));

            }

        }
        

        void loss(  float* probs,
                    int* labels,
                    float* dl_dy,
                    int batch,
                    int n_classes){
        
            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if(i >= batch) return;
    
            for(int c = 0; c < n_classes-1; c++){
                int index = i * n_classes + c;

                if(c == labels[i]){
                    dl_dy[index] = probs[index] - 1.0f;
                
                } else {
                    dl_dy[index] = probs[index];
                }
            }

        }


        void backprop(  Layer& current,
                        float* X,
                        float* dl_dy,
                        float* dl_dx,
                        int batch){
            

            // dL/dX_pre_act = dL/dY * act'(X)
            activation_backp<<<NUM_BLOCKS, BLOCK_SIZE>>>
            (X, 
            dl_dy, dl_dx, 
            batch * current.in_dim, 
            current.activation_type);
 

            // dL/dW = Xᵀ @ dL/dY
            matmul_backp_W<<< NUM_BLOCKS, BLOCK_SIZE>>>
            (X, 
            dl_dy, current.dW, 
            batch, current.in_dim, current.out_dim);
            

            // dL/db = sum over batch of dL/dY
            matmul_backp_b<<<NUM_BLOCKS, BLOCK_SIZE>>>
            (dl_dy, current.db, 
            batch, current.out_dim);
                
            
            // dL/dX = dL/dY @ Wᵀ
            matmul_backp_W<<<NUM_BLOCKS, BLOCK_SIZE>>>
            (current.W, 
            dl_dy, dl_dx,
            batch,
            current.in_dim, current.out_dim);
            
            if(training){
                
                dropout_backprop<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (dl_dy, dl_dx, 
                current.mask, batch * current.in_dim);
                
            }

        }


        void update(Network& network){
                
            Layer current;
            for(int i = 0; i < network.n_layers; i++){
                
                Layer current = network.layers[i];

                adam<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (current.W, current.dW, 
                 current.m_W, current.v_W,
                 network.lr, 
                 network.beta_1, network.beta_2,
                 network.epsilon, network.timestamp,
                 current.out_dim);

                adam<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (current.b, current.db, 
                 current.m_b, current.v_b,
                 network.lr, 
                 network.beta_1, network.beta_2,
                 network.epsilon, network.timestamp,
                 current.out_dim);

            }
        }   
        

        void train( Network& network, Dataset& train, 
                    int epochs, int batch_size){
            
            const int num_batches = ceil(train.num_rows + batch_size-1 / batch_size);
            const int num_activations = network.n_layers + 1;
            float** A = new float*[num_activations];
            float** dA = new float*[num_activations];

                    
                for(int layer = 0; layer < num_activations; layer++){
                    
                    int dim = (layer == 0) ? network.num_features : network.layers[layer-1].out_dim;

                    cudaMalloc(&A[layer], batch_size * dim * sizeof(float));                    
                    cudaMalloc(&dA[layer], batch_size * dim * sizeof(float));                    
                    
                }
                
                    
                
                // ─────────────────────────────────────────────────────────────────────────────────────────────────
                // declaração dos arrays de resultado (y_gender, y_accord)

                int* d_y_gender;
                int* d_y_accord;
                cudaMalloc(&d_y_gender, batch_size * sizeof(int));
                cudaMalloc(&d_y_accord, batch_size * sizeof(int));



                for(int epoch = 0; epoch < epochs-1; epoch++){
                    for(int batch = 0; batch < num_batches; batch++){
                        
                        unsigned long long seed = time(NULL);
                        int current_batch = min(batch_size, train.num_rows - batch * batch_size);
                        cudaMemcpy( A[0], 
                                    train.X + (batch * batch_size * network.num_features),
                                    (current_batch * network.num_features) * sizeof(float),
                                    cudaMemcpyHostToDevice
                                  );


                        // ─────────────────────────────────────────────────────────────────────────────────────────────────
                        // fwd_pass + softmax at the end

                        for(int i = 0; i < network.n_layers-1; i++){
                            fwd_pass(network, network.layers[i], A[i], A[i+1], batch);
                        }


                        // ─────────────────────────────────────────────────────────────────────────────────────────────────
                        // compute loss 
                        loss(A[network.n_layers], d_y_gender, dA[network.n_layers], current_batch, 3);
                        

                        // ─────────────────────────────────────────────────────────────────────────────────────────────────
                        // backprop

                        for(int i = network.n_layers-1; i < 0; i--){
                            backprop(network.layers[i], A[i], dA[i+1], dA[i], current_batch);
                        }
                        

                        // ─────────────────────────────────────────────────────────────────────────────────────────────────
                        // update
                        update(network);
                        network.timestamp++;

                }
            }               

            // ─────────────────────────────────────────────────────────────────────────────────────────────────
            // free memory
            
            for(int i = 0; i < network.n_layers-1; i++){

                cudaFree(A[i]);
                cudaFree(dA[i]);
                
                Layer& L = network.layers[i];

                cudaFree(L.W);
                cudaFree(L.b);

                cudaFree(L.dW);
                cudaFree(L.db);

                cudaFree(L.v_W);
                cudaFree(L.m_W);

                cudaFree(L.v_b);
                cudaFree(L.m_b);

                cudaFree(L.mask);
            }
            
            cudaFree(d_y_gender);
            cudaFree(d_y_accord);

            delete[] A;
            delete[] dA;
            delete[] network.layers;
            
            free_data(train);
        }
        

        void write_weights(){}

        
        void test(){}


}; 


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
