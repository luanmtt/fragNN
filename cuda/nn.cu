#include "kernel.cuh"
#include "../src/headers/dataloader.hpp"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct Layer {

    int in_dim; 
    int out_dim;
    int activation_type; // from ActivationType enum
    
    float* W;   // in * out
    float* b;   // out

    float* dW;  // in * out
    float* db;  // out

    float* m_W;
    float* v_W;
    float* m_b;
    float* v_b;

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

};



class NeuralNetwork {
    
    private:

        int NUM_BLOCKS = 16;
        int BLOCK_SIZE = 16;
        bool training;

    public:
        

        Network create_network( int* layers_dims,
                                int n_layers,
                                float lr,
                                float keep_prob){
                
            for(int i = 0; i < n_layers; i++){
            


            }
        }
        

        void fwd_pass(  Layer& current, 
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
                (Y, float *mask,  
                keep_prob,  batch * current.in_dim, seed);

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
                float *mask, batch * current.in_dim);
                
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
        
            
            for(int epoch = 0; epoch < epochs-1; epoch++){

                
                for(int batch = 0; batch < batch_size; batch++){

                }

            }

        }

}; 



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
