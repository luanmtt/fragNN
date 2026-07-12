#include "kernel.cuh"
#include "../src/headers/dataloader.hpp"

#include <ctime>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// free __global__ kernels — cannot be class methods


__global__ void loss_kernel(    float* probs,
                                int* labels,
                                float* dl_dy,
                                int batch,
                                int n_classes){

    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= batch) return;

    for(int c = 0; c < n_classes; c++){
        int index = i * n_classes + c;

        if(c == labels[i]){
            dl_dy[index] = probs[index] - 1.0f;
        } else {
            dl_dy[index] = probs[index];
        }
    }
}


__global__ void sum_gradients(float* dst, float* src, int n){

    /*
        // merge gradients from two heads into shared backbone
        // dst[i] = dst[i] + src[i]
    */

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;

    dst[i] += src[i];
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


struct Layer {

    int in_dim; 
    int out_dim;
    int activation_type; // ActivationType enum, or -1 for no activation
    
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
    
    Layer* layers;          // shared backbone [0..n_shared-1]
    int n_shared;

    Layer gender_head;      // 64 → 3
    Layer accord_head;      // 64 → 84

    // hyperparams
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

        void _set_NUM_BLOCKS_(int n){
            NUM_BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }

        void _alloc_layer_(Layer& L, int in_dim, int out_dim, int act){
            L.in_dim = in_dim;
            L.out_dim = out_dim;
            L.activation_type = act;

            int n_W = in_dim * out_dim;
            int n_b = out_dim;

            cudaMalloc(&L.W,    n_W * sizeof(float));
            cudaMalloc(&L.b,    n_b * sizeof(float));
            cudaMalloc(&L.dW,   n_W * sizeof(float));
            cudaMalloc(&L.db,   n_b * sizeof(float));
            cudaMalloc(&L.m_W,  n_W * sizeof(float));
            cudaMalloc(&L.v_W,  n_W * sizeof(float));
            cudaMalloc(&L.m_b,  n_b * sizeof(float));
            cudaMalloc(&L.v_b,  n_b * sizeof(float));
            cudaMalloc(&L.mask, n_b * sizeof(float));

            _set_NUM_BLOCKS_(n_W);
            HE<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.W, n_W, in_dim);
            init_b<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.b, n_b);

            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.dW,   n_W);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.db,   n_b);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.m_W,  n_W);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.v_W,  n_W);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.m_b,  n_b);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.v_b,  n_b);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.mask, n_b);
        }

        void _free_layer_(Layer& L){
            cudaFree(L.W);    cudaFree(L.b);
            cudaFree(L.dW);   cudaFree(L.db);
            cudaFree(L.m_W);  cudaFree(L.v_W);
            cudaFree(L.m_b);  cudaFree(L.v_b);
            cudaFree(L.mask);
        }

    public:

        Network create_network( int* shared_dims,
                                int n_shared,
                                int n_gender,
                                int n_accord,
                                int middle_act,
                                float lr,
                                float keep_prob){

            Network net;
            net.keep_prob = keep_prob;
            net.lr = lr;
            net.n_shared = n_shared;
            net.timestamp = 0;
            net.num_features = shared_dims[0];

            net.beta_1   = 0.9f;
            net.beta_2   = 0.999f;
            net.epsilon  = 1e-8f;

            // shared backbone
            net.layers = new Layer[n_shared];
            for(int i = 0; i < n_shared; i++){
                _alloc_layer_(net.layers[i], shared_dims[i], shared_dims[i+1], middle_act);
            }

            // output heads — no activation (softmax applied separately)
            int split_dim = shared_dims[n_shared];
            _alloc_layer_(net.gender_head, split_dim, n_gender, -1);
            _alloc_layer_(net.accord_head, split_dim, n_accord, -1);

            return net;
        }


        void fwd_pass(Layer& current, float* X, float* Y, int batch){

            _set_NUM_BLOCKS_(current.in_dim * current.out_dim);

            matmul<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                X, Y, current.W, current.b,
                batch, current.in_dim, current.out_dim);

            if(current.activation_type >= 0){
                apply_activation<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                    Y, batch * current.out_dim, current.activation_type);

                if(training){
                    dropout_forward<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        Y, current.mask, keep_prob,
                        batch * current.out_dim, (unsigned long long)timestamp * batch);
                }
            }
        }


        void backprop(  Layer& current,
                        float* X,
                        float* dl_dy,
                        float* dl_dx,
                        int batch){

            _set_NUM_BLOCKS_(current.in_dim * current.out_dim);

            // activation backward (skip for output heads — no activation applied)
            if(current.activation_type >= 0){
                activation_backp<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                    X, dl_dy, dl_dx,
                    batch * current.in_dim, current.activation_type);
            }

            matmul_backp_W<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                X, dl_dy, current.dW,
                batch, current.in_dim, current.out_dim);

            matmul_backp_b<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                dl_dy, current.db,
                batch, current.out_dim);

            matmul_backp_X<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                current.W, dl_dy, dl_dx,
                batch, current.in_dim, current.out_dim);

            if(training && current.activation_type >= 0){
                dropout_backprop<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                    dl_dy, dl_dx, current.mask, batch * current.in_dim);
            }
        }


        void train( Network& net, Dataset& train,
                    int epochs, int batch_size){

            training = true;

            const int num_batches = (train.num_rows + batch_size - 1) / batch_size;
            const int n_shared    = net.n_shared;

            // A[0] = input, A[1..n_shared] = shared layer outputs
            // A[n_shared+1] = gender output, A[n_shared+2] = accord output
            float** A  = new float*[n_shared + 3];
            float** dA = new float*[n_shared + 3];

            for(int i = 0; i <= n_shared; i++){
                int dim = (i == 0) ? net.num_features : net.layers[i-1].out_dim;
                cudaMalloc(&A[i],  batch_size * dim * sizeof(float));
                cudaMalloc(&dA[i], batch_size * dim * sizeof(float));
            }

            int gid = n_shared + 1;  // gender output index
            int aid = n_shared + 2;  // accord output index

            cudaMalloc(&A[gid],  batch_size * 3  * sizeof(float));
            cudaMalloc(&A[aid],  batch_size * 84 * sizeof(float));
            cudaMalloc(&dA[gid], batch_size * 3  * sizeof(float));
            cudaMalloc(&dA[aid], batch_size * 84 * sizeof(float));

            int* d_y_gender;
            int* d_y_accord;
            cudaMalloc(&d_y_gender, batch_size * sizeof(int));
            cudaMalloc(&d_y_accord, batch_size * sizeof(int));

            for(int epoch = 0; epoch < epochs; epoch++){
                for(int b = 0; b < num_batches; b++){

                    int current_batch = min(batch_size, train.num_rows - b * batch_size);
                    int n_feat = net.num_features;

                    cudaMemcpy(A[0],
                               train.X + b * batch_size * n_feat,
                               current_batch * n_feat * sizeof(float),
                               cudaMemcpyHostToDevice);

                    cudaMemcpy(d_y_gender,
                               train.y_gender + b * batch_size,
                               current_batch * sizeof(int),
                               cudaMemcpyHostToDevice);

                    cudaMemcpy(d_y_accord,
                               train.y_accord + b * batch_size,
                               current_batch * sizeof(int),
                               cudaMemcpyHostToDevice);

                    // ═══ FORWARD — shared backbone ═══
                    for(int i = 0; i < n_shared; i++){
                        fwd_pass(net.layers[i], A[i], A[i+1], current_batch);
                    }

                    // ═══ FORWARD — gender head ═══
                    fwd_pass(net.gender_head, A[n_shared], A[gid], current_batch);
                    softmax<<<(current_batch + 255) / 256, 256>>>(
                        A[gid], current_batch, 3);

                    // ═══ FORWARD — accord head ═══
                    fwd_pass(net.accord_head, A[n_shared], A[aid], current_batch);
                    softmax<<<(current_batch + 255) / 256, 256>>>(
                        A[aid], current_batch, 84);

                    // ═══ LOSS — gender ═══
                    loss_kernel<<<(current_batch + 255) / 256, 256>>>(
                        A[gid], d_y_gender, dA[gid], current_batch, 3);

                    // ═══ LOSS — accord ═══
                    loss_kernel<<<(current_batch + 255) / 256, 256>>>(
                        A[aid], d_y_accord, dA[aid], current_batch, 84);

                    // ═══ BACKWARD — heads ═══
                    backprop(net.gender_head, A[n_shared],
                             dA[gid], dA[n_shared], current_batch);

                    backprop(net.accord_head, A[n_shared],
                             dA[aid], dA[n_shared], current_batch);

                    // ═══ BACKWARD — shared backbone ═══
                    for(int i = n_shared - 1; i >= 0; i--){
                        backprop(net.layers[i], A[i], dA[i+1], dA[i], current_batch);
                    }

                    // ═══ UPDATE — shared backbone ═══
                    for(int i = 0; i < n_shared; i++){
                        _set_NUM_BLOCKS_(net.layers[i].in_dim * net.layers[i].out_dim);
                        adam<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            net.layers[i].W, net.layers[i].dW,
                            net.layers[i].m_W, net.layers[i].v_W,
                            net.lr, net.beta_1, net.beta_2,
                            net.epsilon, net.timestamp,
                            net.layers[i].in_dim * net.layers[i].out_dim);
                        adam<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            net.layers[i].b, net.layers[i].db,
                            net.layers[i].m_b, net.layers[i].v_b,
                            net.lr, net.beta_1, net.beta_2,
                            net.epsilon, net.timestamp,
                            net.layers[i].out_dim);
                    }

                    // ═══ UPDATE — gender head ═══
                    _set_NUM_BLOCKS_(net.gender_head.in_dim * net.gender_head.out_dim);
                    adam<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        net.gender_head.W, net.gender_head.dW,
                        net.gender_head.m_W, net.gender_head.v_W,
                        net.lr, net.beta_1, net.beta_2,
                        net.epsilon, net.timestamp,
                        net.gender_head.in_dim * net.gender_head.out_dim);
                    adam<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        net.gender_head.b, net.gender_head.db,
                        net.gender_head.m_b, net.gender_head.v_b,
                        net.lr, net.beta_1, net.beta_2,
                        net.epsilon, net.timestamp,
                        net.gender_head.out_dim);

                    // ═══ UPDATE — accord head ═══
                    _set_NUM_BLOCKS_(net.accord_head.in_dim * net.accord_head.out_dim);
                    adam<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        net.accord_head.W, net.accord_head.dW,
                        net.accord_head.m_W, net.accord_head.v_W,
                        net.lr, net.beta_1, net.beta_2,
                        net.epsilon, net.timestamp,
                        net.accord_head.in_dim * net.accord_head.out_dim);
                    adam<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        net.accord_head.b, net.accord_head.db,
                        net.accord_head.m_b, net.accord_head.v_b,
                        net.lr, net.beta_1, net.beta_2,
                        net.epsilon, net.timestamp,
                        net.accord_head.out_dim);

                    net.timestamp++;
                }
            }

            // ── free memory ──
            for(int i = 0; i <= aid; i++){
                cudaFree(A[i]);
                cudaFree(dA[i]);
            }
            for(int i = 0; i < n_shared; i++){
                _free_layer_(net.layers[i]);
            }
            _free_layer_(net.gender_head);
            _free_layer_(net.accord_head);

            cudaFree(d_y_gender);
            cudaFree(d_y_accord);
            delete[] A;
            delete[] dA;
            delete[] net.layers;
        }


        void write_weights(Network& net, const char* path){

            FILE* f = fopen(path, "wb");

            fwrite(&net.n_shared, sizeof(int), 1, f);

            for(int i = 0; i < net.n_shared; i++){
                Layer& L = net.layers[i];
                fwrite(&L.in_dim,  sizeof(int), 1, f);
                fwrite(&L.out_dim, sizeof(int), 1, f);
                int n_W = L.in_dim * L.out_dim;
                float* h_W = new float[n_W];
                float* h_b = new float[L.out_dim];
                cudaMemcpy(h_W, L.W, n_W * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_b, L.b, L.out_dim * sizeof(float), cudaMemcpyDeviceToHost);
                fwrite(h_W, sizeof(float), n_W, f);
                fwrite(h_b, sizeof(float), L.out_dim, f);
                delete[] h_W;
                delete[] h_b;
            }

            // gender head
            fwrite(&net.gender_head.in_dim,  sizeof(int), 1, f);
            fwrite(&net.gender_head.out_dim, sizeof(int), 1, f);
            {
                int n_W = net.gender_head.in_dim * net.gender_head.out_dim;
                float* h_W = new float[n_W];
                float* h_b = new float[net.gender_head.out_dim];
                cudaMemcpy(h_W, net.gender_head.W, n_W * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_b, net.gender_head.b, net.gender_head.out_dim * sizeof(float), cudaMemcpyDeviceToHost);
                fwrite(h_W, sizeof(float), n_W, f);
                fwrite(h_b, sizeof(float), net.gender_head.out_dim, f);
                delete[] h_W;
                delete[] h_b;
            }

            // accord head
            fwrite(&net.accord_head.in_dim,  sizeof(int), 1, f);
            fwrite(&net.accord_head.out_dim, sizeof(int), 1, f);
            {
                int n_W = net.accord_head.in_dim * net.accord_head.out_dim;
                float* h_W = new float[n_W];
                float* h_b = new float[net.accord_head.out_dim];
                cudaMemcpy(h_W, net.accord_head.W, n_W * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_b, net.accord_head.b, net.accord_head.out_dim * sizeof(float), cudaMemcpyDeviceToHost);
                fwrite(h_W, sizeof(float), n_W, f);
                fwrite(h_b, sizeof(float), net.accord_head.out_dim, f);
                delete[] h_W;
                delete[] h_b;
            }

            fclose(f);
        }


        void load_weights(Network& net, const char* path){

            FILE* f = fopen(path, "rb");

            int n_shared;
            fread(&n_shared, sizeof(int), 1, f);

            for(int i = 0; i < n_shared; i++){
                int in_dim, out_dim;
                fread(&in_dim,  sizeof(int), 1, f);
                fread(&out_dim, sizeof(int), 1, f);
                int n_W = in_dim * out_dim;
                float* h_W = new float[n_W];
                float* h_b = new float[out_dim];
                fread(h_W, sizeof(float), n_W, f);
                fread(h_b, sizeof(float), out_dim, f);
                cudaMemcpy(net.layers[i].W, h_W, n_W * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(net.layers[i].b, h_b, out_dim * sizeof(float), cudaMemcpyHostToDevice);
                delete[] h_W;
                delete[] h_b;
            }

            // gender head
            {
                int in_dim, out_dim;
                fread(&in_dim,  sizeof(int), 1, f);
                fread(&out_dim, sizeof(int), 1, f);
                int n_W = in_dim * out_dim;
                float* h_W = new float[n_W];
                float* h_b = new float[out_dim];
                fread(h_W, sizeof(float), n_W, f);
                fread(h_b, sizeof(float), out_dim, f);
                cudaMemcpy(net.gender_head.W, h_W, n_W * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(net.gender_head.b, h_b, out_dim * sizeof(float), cudaMemcpyHostToDevice);
                delete[] h_W;
                delete[] h_b;
            }

            // accord head
            {
                int in_dim, out_dim;
                fread(&in_dim,  sizeof(int), 1, f);
                fread(&out_dim, sizeof(int), 1, f);
                int n_W = in_dim * out_dim;
                float* h_W = new float[n_W];
                float* h_b = new float[out_dim];
                fread(h_W, sizeof(float), n_W, f);
                fread(h_b, sizeof(float), out_dim, f);
                cudaMemcpy(net.accord_head.W, h_W, n_W * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(net.accord_head.b, h_b, out_dim * sizeof(float), cudaMemcpyHostToDevice);
                delete[] h_W;
                delete[] h_b;
            }

            fclose(f);
        }

}; 


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
