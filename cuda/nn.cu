#include "kernel.cuh"
#include "../src/headers/dataloader.hpp"
#include "../utils/embellish.cuh"

#include <ctime>
#include <cstdio>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// __global__ kernels — não podem ser métodos de classe


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
        // mistura gradientes de duas classificações num backbone só
        // dst[i] = dst[i] + src[i]
    */

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) return;

    dst[i] += src[i];
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// wrappers — chamáveis de main.cpp (C++ puro, sem <<<>>>


void launch_loss_kernel(    float* probs,
                            int* labels,
                            float* dl_dy,
                            int batch,
                            int n_classes){

    int blocks = (batch + 255) / 256;
    loss_kernel<<<blocks, 256>>>(probs, labels, dl_dy, batch, n_classes);
}


void launch_sum_gradients(  float* dst,
                            float* src,
                            int n){

    int blocks = (n + 255) / 256;
    sum_gradients<<<blocks, 256>>>(dst, src, n);
}


float compute_batch_loss(float* h_probs, int* h_labels, int batch, int n_classes){

    /*
        calcula a loss média (cross-entropy) de um batch no CPU.
        fórmula: loss = -mean(log(probs[correct_class]))

        variáveis:
            h_probs   — probabilidades do batch (batch × n_classes), já no CPU
            h_labels  — rótulos verdadeiros (batch)
            batch     — número de amostras
            n_classes — número de classes

        exemplo: batch=2, n_classes=3
            h_probs  = [0.1, 0.7, 0.2,   // amostra 0: predito classe 1
                         0.8, 0.1, 0.1]   // amostra 1: predito classe 0
            h_labels = [1, 0]

            loss = -(log(0.7) + log(0.8)) / 2 = (0.357 + 0.223) / 2 = 0.290
    */

    const float eps = 1e-15f;
    float total_loss = 0.0f;

    for(int i = 0; i < batch; i++){
        int label = h_labels[i];
        float prob = h_probs[i * n_classes + label];
        total_loss += -logf(prob + eps);
    }

    return total_loss / (float)batch;
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

            int num_W = in_dim * out_dim;
            int num_b = out_dim;

            cudaMalloc(&L.W,    num_W * sizeof(float));
            cudaMalloc(&L.b,    num_b * sizeof(float));
            cudaMalloc(&L.dW,   num_W * sizeof(float));
            cudaMalloc(&L.db,   num_b * sizeof(float));
            cudaMalloc(&L.m_W,  num_W * sizeof(float));
            cudaMalloc(&L.v_W,  num_W * sizeof(float));
            cudaMalloc(&L.m_b,  num_b * sizeof(float));
            cudaMalloc(&L.v_b,  num_b * sizeof(float));
            cudaMalloc(&L.mask, num_b * sizeof(float));

            _set_NUM_BLOCKS_(num_W);
            HE<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.W, num_W, in_dim);
            init_b<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.b, num_b);

            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.dW,   num_W);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.db,   num_b);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.m_W,  num_W);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.v_W,  num_W);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.m_b,  num_b);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.v_b,  num_b);
            zeroGradients<<<NUM_BLOCKS, BLOCK_SIZE>>>(L.mask, num_b);
        }

        void _free_layer_(Layer& L){
            cudaFree(L.W);    cudaFree(L.b);
            cudaFree(L.dW);   cudaFree(L.db);
            cudaFree(L.m_W);  cudaFree(L.v_W);
            cudaFree(L.m_b);  cudaFree(L.v_b);
            cudaFree(L.mask);
        }

        void fwd_pass(Network& network, Layer& current, float* X, float* Y, int batch){

            _set_NUM_BLOCKS_(current.in_dim * current.out_dim);

            matmul<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                X, Y, current.W, current.b,
                batch, current.in_dim, current.out_dim);

            if(current.activation_type >= 0){
                apply_activation<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                    Y, batch * current.out_dim, current.activation_type);

                if(training){
                    dropout_forward<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        Y, current.mask, network.keep_prob,
                        batch * current.out_dim, (unsigned long long)network.timestamp * batch);
                }
            }
        }


        void backprop(  Layer& current,
                        float* X,
                        float* dl_dy,
                        float* dl_dx,
                        int batch){

            _set_NUM_BLOCKS_(current.in_dim * current.out_dim);

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

    public:

        static const int GENDER_SIZE = 3;
        static const int ACCORD_SIZE = 84;

        metrics_used metrics_t;

        Network create_network( int* shared_dims,
                                int n_shared,
                                int n_gender,
                                int n_accord,
                                int middle_act,
                                float lr,
                                float keep_prob);

        void train( Network& net, Dataset& train,
                    int epochs, int batch_size);

        void test(  Network& net, Dataset& test_data,
                    int batch_size,
                    int verbosity = 1);

        void write_weights(Network& net, const char* path);

        void load_weights(Network& net, const char* path);

        void destroy_network(Network& net);

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// implementação dos métodos públicos


Network NeuralNetwork::create_network(   int* shared_dims,
                                        int n_shared,
                                        int n_gender,
                                        int n_accord,
                                        int middle_act,
                                        float lr,
                                        float keep_prob){

    /*
        cria e inicializa a rede neural com duas cabeças de saída.

        variáveis:
            shared_dims  — dimensões do backbone compartilhado. ex: {128, 64, 32}
                           o primeiro elemento é num_features (entrada).
            n_shared     — número de camadas compartilhadas. ex: 2 para {128, 64, 32}
            n_gender     — número de classes de gênero. ex: 3 (masculino, feminino, unissex)
            n_accord     — número de classes de acordes. ex: 84
            middle_act   — tipo de ativação para camadas intermediárias (ActivationType enum)
            lr           — taxa de aprendizado (learning rate). ex: 0.001
            keep_prob    — probabilidade de manter um neurônio no dropout. ex: 0.9

        exemplo:
            int dims[] = {num_features, 128, 64};
            Network net = create_network(dims, 2, 3, 84, ACT_LEAKY_RELU, 0.001f, 0.9f);

            resultado:
                backbone: num_features → 128 (LEAKY_RELU) → 64 (LEAKY_RELU)
                gender_head: 64 → 3 (sem ativação — softmax separado)
                accord_head: 64 → 84 (sem ativação — softmax separado)
    */

    Network net;
    net.keep_prob = keep_prob;
    net.lr = lr;
    net.n_shared = n_shared;
    net.timestamp = 1;
    net.num_features = shared_dims[0];


    // ─────────────────────────────────────────────────────────────────────────────────────────────────
    // magic numbers
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


void NeuralNetwork::train(  Network& net, Dataset& train,
                            int epochs, int batch_size){

    /*
        treina a rede neural com duas cabeças (gênero e acordes).

        variáveis:
            net          — rede neural com backbone compartilhado + gender_head + accord_head
            train        — conjunto de dados de treino (CPU), contém X, y_gender, y_accord
            epochs       — número de passagens completas sobre os dados. ex: 50
            batch_size   — número de amostras por mini-batch. ex: 32

            num_batches  — total de batches por época = ceil(num_rows / batch_size)
            n_shared     — número de camadas compartilhadas no backbone
            A[i]         — buffers de ativações na GPU. A[0]=entrada, A[1..n_shared]=backbone,
                           A[gid]=saída gênero, A[aid]=saída acordes
            dA[i]        — buffers de gradientes na GPU (mesmos índices de A)
            gid          — índice do buffer de gênero em A[] (= n_shared + 1)
            aid          — índice do buffer de acordes em A[] (= n_shared + 2)
            d_y_gender   — rótulos de gênero do batch atual na GPU
            d_y_accord   — rótulos de acordes do batch atual na GPU

        fluxo por batch:
            1. copiar entrada + rótulos para GPU
            2. forward backbone compartilhado
            3. forward gender_head → softmax
            4. forward accord_head → softmax
            5. loss_kernel para gênero → dA[gid]
            6. loss_kernel para acordes → dA[aid]
            7. backward ambos os heads (acumula gradientes em dA[n_shared])
            8. backward backbone compartilhado
            9. adam update em todos os pesos
    */

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

    cudaMalloc(&A[gid],  batch_size * GENDER_SIZE  * sizeof(float));
    cudaMalloc(&A[aid],  batch_size * ACCORD_SIZE * sizeof(float));
    cudaMalloc(&dA[gid], batch_size * GENDER_SIZE  * sizeof(float));
    cudaMalloc(&dA[aid], batch_size * ACCORD_SIZE * sizeof(float));

    int* d_y_gender;
    int* d_y_accord;
    cudaMalloc(&d_y_gender, batch_size * sizeof(int));
    cudaMalloc(&d_y_accord, batch_size * sizeof(int));

    for(int epoch = 0; epoch < epochs; epoch++){
        for(int b = 0; b < num_batches; b++){

            int current_batch = min(batch_size, train.num_rows - b * batch_size);
            int n_feat = net.num_features;

            _set_NUM_BLOCKS_(current_batch);

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
                fwd_pass(net, net.layers[i], A[i], A[i+1], current_batch);
            }

            // ═══ FORWARD — gender head ═══
            fwd_pass(net, net.gender_head, A[n_shared], A[gid], current_batch);
            softmax<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                A[gid], current_batch, GENDER_SIZE);

            // ═══ FORWARD — accord head ═══
            fwd_pass(net, net.accord_head, A[n_shared], A[aid], current_batch);
            softmax<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                A[aid], current_batch, ACCORD_SIZE);

            // ═══ LOSS computation (CPU) ═══
            float* h_probs_gender = new float[current_batch * GENDER_SIZE];
            float* h_probs_accord = new float[current_batch * ACCORD_SIZE];
            int* h_labels_gender = new int[current_batch];
            int* h_labels_accord = new int[current_batch];

            cudaMemcpy(h_probs_gender, A[gid], current_batch * GENDER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_probs_accord, A[aid], current_batch * ACCORD_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_labels_gender, d_y_gender, current_batch * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_labels_accord, d_y_accord, current_batch * sizeof(int), cudaMemcpyDeviceToHost);

            float loss_gender = compute_batch_loss(h_probs_gender, h_labels_gender, current_batch, GENDER_SIZE);
            float loss_accord = compute_batch_loss(h_probs_accord, h_labels_accord, current_batch, ACCORD_SIZE);
            float loss_total = loss_gender + loss_accord;

            delete[] h_probs_gender;
            delete[] h_probs_accord;
            delete[] h_labels_gender;
            delete[] h_labels_accord;

            // ═══ LOSS — gender ═══
            loss_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                A[gid], d_y_gender, dA[gid], current_batch, GENDER_SIZE);

            // ═══ LOSS — accord ═══
            loss_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                A[aid], d_y_accord, dA[aid], current_batch, ACCORD_SIZE);

            // ═══ BACKWARD — heads ═══
            // gender head: dA[gid] → dA[n_shared]
            backprop(net.gender_head, A[n_shared],
                     dA[gid], dA[n_shared], current_batch);

            // accord head: dA[aid] → dA_temp (buffer temporário)
            float* dA_temp;
            int split_dim = net.layers[n_shared - 1].out_dim;
            cudaMalloc(&dA_temp, current_batch * split_dim * sizeof(float));

            backprop(net.accord_head, A[n_shared],
                     dA[aid], dA_temp, current_batch);

            // soma gradientes: dA[n_shared] += dA_temp
            sum_gradients<<<(current_batch * split_dim + 255) / 256, 256>>>(
                dA[n_shared], dA_temp, current_batch * split_dim);

            cudaFree(dA_temp);

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

            // ═══ PROGRESS — epoch display ═══
            float progress = (float)(b + 1) / (float)num_batches * 100.0f;

            if(b == 0){
                std::string epoch_label = std::string("Epoch ") + std::to_string(epoch + 1) + "/" + std::to_string(epochs);
                std::string bar = sepBarComment(60, stylePrint(epoch_label.c_str(), "#78a9c7", true).c_str());
                printf("%s", bar.c_str());
            }

            printf("\r  batch %d/%d | loss: %.4f | lr: %.4f | %.1f%%   ",
                b + 1, num_batches, loss_total, net.lr, progress);
            fflush(stdout);

            if(b == num_batches - 1) printf("\n");
        }
    }

    // ── free memory ──
    for(int i = 0; i <= aid; i++){
        cudaFree(A[i]);
        cudaFree(dA[i]);
    }

    cudaFree(d_y_gender);
    cudaFree(d_y_accord);
    delete[] A;
    delete[] dA;
}


void NeuralNetwork::write_weights(Network& net, const char* path){

    /*
        salva os pesos treinados em um arquivo binário.
        usado para carregar o modelo depois sem precisar treinar novamente.

        variáveis:
            net   — rede neural com pesos treinados na GPU
            path  — caminho do arquivo de saída. ex: "weights.bin"

        formato do arquivo:
            [n_shared]
            [in_dim_0, out_dim_0, W_0..., b_0...]  (camada compartilhada 0)
            [in_dim_1, out_dim_1, W_1..., b_1...]  (camada compartilhada 1)
            ...
            [in_dim_gender, out_dim_gender, W_gender..., b_gender...]
            [in_dim_accord, out_dim_accord, W_accord..., b_accord...]

        exemplo: backbone {128, 64}, gender 64→3, accord 64→84
            arquivo contém:
                2                                    (n_shared)
                128, 64, [128×64 floats], [64 floats]  (camada 0)
                64, 64,  [64×64 floats],  [64 floats]  (camada 1)
                64, 3,   [64×3 floats],   [3 floats]   (gender head)
                64, 84,  [64×84 floats],  [84 floats]  (accord head)
    */

    FILE* f = fopen(path, "wb");

    fwrite(&net.n_shared, sizeof(int), 1, f);

    for(int i = 0; i < net.n_shared; i++){
        Layer& L = net.layers[i];

        fwrite(&L.in_dim,  sizeof(int), 1, f);
        fwrite(&L.out_dim, sizeof(int), 1, f);

        int num_W = L.in_dim * L.out_dim;
        float* result_W = new float[num_W];
        float* result_b = new float[L.out_dim];

        cudaMemcpy(result_W, L.W, num_W * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_b, L.b, L.out_dim * sizeof(float), cudaMemcpyDeviceToHost);

        fwrite(result_W, sizeof(float), num_W, f);
        fwrite(result_b, sizeof(float), L.out_dim, f);

        delete[] result_W;
        delete[] result_b;
    }

    // gender head
    fwrite(&net.gender_head.in_dim,  sizeof(int), 1, f);
    fwrite(&net.gender_head.out_dim, sizeof(int), 1, f);
    {
        int num_W = net.gender_head.in_dim * net.gender_head.out_dim;

        float* result_W = new float[num_W];
        float* result_b = new float[net.gender_head.out_dim];

        cudaMemcpy(result_W, net.gender_head.W, num_W * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_b, net.gender_head.b, net.gender_head.out_dim * sizeof(float), cudaMemcpyDeviceToHost);

        fwrite(result_W, sizeof(float), num_W, f);
        fwrite(result_b, sizeof(float), net.gender_head.out_dim, f);

        delete[] result_W;
        delete[] result_b;
    }

    // accord head
    fwrite(&net.accord_head.in_dim,  sizeof(int), 1, f);
    fwrite(&net.accord_head.out_dim, sizeof(int), 1, f);
    {
        int num_W = net.accord_head.in_dim * net.accord_head.out_dim;

        float* result_W = new float[num_W];
        float* result_b = new float[net.accord_head.out_dim];

        cudaMemcpy(result_W, net.accord_head.W, num_W * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_b, net.accord_head.b, net.accord_head.out_dim * sizeof(float), cudaMemcpyDeviceToHost);

        fwrite(result_W, sizeof(float), num_W, f);
        fwrite(result_b, sizeof(float), net.accord_head.out_dim, f);

        delete[] result_W;
        delete[] result_b;
    }

    fclose(f);
}


void NeuralNetwork::load_weights(Network& net, const char* path){

    /*
        carrega pesos de um arquivo binário para a rede neural na GPU.
        usado para inference sem precisar treinar.

        variáveis:
            net   — rede neural já criada com create_network (pesos serão sobrescritos)
            path  — caminho do arquivo binário. ex: "weights.bin"

        fluxo:
            1. ler n_shared do arquivo
            2. para cada camada compartilhada: ler dimensões, W e b → copiar para GPU
            3. ler gender head: dimensões, W e b → copiar para GPU
            4. ler accord head: dimensões, W e b → copiar para GPU

        nota: a rede deve ser criada com as mesmas dimensões do arquivo.
        se as dimensões não baterem, os dados serão escritos incorretamente (sem validação).
    */

    FILE* f = fopen(path, "rb");

    int num_shared;
    fread(&num_shared, sizeof(int), 1, f);

    for(int i = 0; i < num_shared; i++){

        int inum_dim, out_dim;

        fread(&inum_dim,  sizeof(int), 1, f);
        fread(&out_dim, sizeof(int), 1, f);

        int num_W = inum_dim * out_dim;

        float* subject_W = new float[num_W];
        float* subject_b = new float[out_dim];

        fread(subject_W, sizeof(float), num_W, f);
        fread(subject_b, sizeof(float), out_dim, f);

        cudaMemcpy(net.layers[i].W, subject_W, num_W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(net.layers[i].b, subject_b, out_dim * sizeof(float), cudaMemcpyHostToDevice);

        delete[] subject_W;
        delete[] subject_b;
    }

    // gender head
    {
        int inum_dim, out_dim;

        fread(&inum_dim,  sizeof(int), 1, f);
        fread(&out_dim, sizeof(int), 1, f);

        int num_W = inum_dim * out_dim;

        float* subject_W = new float[num_W];
        float* subject_b = new float[out_dim];

        fread(subject_W, sizeof(float), num_W, f);
        fread(subject_b, sizeof(float), out_dim, f);

        cudaMemcpy(net.gender_head.W, subject_W, num_W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(net.gender_head.b, subject_b, out_dim * sizeof(float), cudaMemcpyHostToDevice);

        delete[] subject_W;
        delete[] subject_b;
    }

    // accord head
    {
        int inum_dim, out_dim;

        fread(&inum_dim,  sizeof(int), 1, f);
        fread(&out_dim, sizeof(int), 1, f);

        int num_W = inum_dim * out_dim;

        float* subject_W = new float[num_W];
        float* subject_b = new float[out_dim];

        fread(subject_W, sizeof(float), num_W, f);
        fread(subject_b, sizeof(float), out_dim, f);

        cudaMemcpy(net.accord_head.W, subject_W, num_W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(net.accord_head.b, subject_b, out_dim * sizeof(float), cudaMemcpyHostToDevice);

        delete[] subject_W;
        delete[] subject_b;
    }

    fclose(f);
}


void NeuralNetwork::test(   Network& net, Dataset& test_data,
                            int batch_size,
                            int verbosity){

    /*
        test(): executa o forward pass nos dados de teste e calcula métricas.
        não há backward pass nem update — apenas predição e avaliação.

        verbosity:
            0 = silencioso (só métricas macro)
            1 = imprime per-class + AUC (padrão)
            2 = imprime tudo + gera CSVs
    */

    // ── validação ──
    if(test_data.num_rows <= 0 || batch_size <= 0){
        printf("[erro] test: num_rows=%d, batch_size=%d — parâmetros inválidos\n",
               test_data.num_rows, batch_size);
        return;
    }

    training = false;
    const int num_batches = (test_data.num_rows + batch_size - 1) / batch_size;
    const int n_shared = net.n_shared;

    const int gid = n_shared + 1;  // índice do buffer de saída de gênero em A[]
    const int aid = n_shared + 2;  // índice do buffer de saída de acordes em A[]

    // ── alocar buffers de ativação na GPU (só A[], sem dA[]) ──
    float** A = new float*[n_shared + 3];

    for(int i = 0; i <= n_shared; i++){
        int dim = (i == 0) ? net.num_features : net.layers[i-1].out_dim;
        cudaMalloc(&A[i],  batch_size * dim * sizeof(float));
    }

    cudaMalloc(&A[gid],  batch_size * GENDER_SIZE  * sizeof(float));
    cudaMalloc(&A[aid],  batch_size * ACCORD_SIZE * sizeof(float));

    // ── buffers CPU para receber predições de cada batch ──
    float* result_gender = new float[batch_size * GENDER_SIZE];
    float* result_accord = new float[batch_size * ACCORD_SIZE];

    // ── buffers CPU para acumular TODAS as predições (necessário para ROC) ──
    float* all_probs_gender = new float[test_data.num_rows * GENDER_SIZE];
    float* all_probs_accord = new float[test_data.num_rows * ACCORD_SIZE];

    // ── rótulos verdadeiros de todas as amostras ──
    int* labels_gender = new int[test_data.num_rows];
    int* labels_accord = new int[test_data.num_rows];

    // copiar rótulos do dataset para os arrays locais
    for(int i = 0; i < test_data.num_rows; i++){
        labels_gender[i] = test_data.y_gender[i];
        labels_accord[i] = test_data.y_accord[i];
    }

    // ── inicializar matrizes de confusão (FORA do loop de batches) ──
    Confusion conf_gender, conf_accord;
    confusion_init(conf_gender, GENDER_SIZE);
    confusion_init(conf_accord, ACCORD_SIZE);

    for(int b = 0; b < num_batches; b++){

        int current_batch = min(batch_size, test_data.num_rows - b * batch_size);

        _set_NUM_BLOCKS_(current_batch);

        // copiar batch de entrada para GPU
        cudaMemcpy(A[0],
                   test_data.X + b * batch_size * net.num_features,
                   current_batch * net.num_features * sizeof(float),
                   cudaMemcpyHostToDevice);

        // ── forward — backbone compartilhado ──
        for(int i = 0; i < n_shared; i++){
            fwd_pass(net, net.layers[i], A[i], A[i+1], current_batch);
        }

        // ── forward — cabeça de gênero + softmax ──
        fwd_pass(net, net.gender_head, A[n_shared], A[gid], current_batch);
            softmax<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                A[gid], current_batch, GENDER_SIZE);

        // ── forward — cabeça de acordes + softmax ──
        fwd_pass(net, net.accord_head, A[n_shared], A[aid], current_batch);
            softmax<<<NUM_BLOCKS, BLOCK_SIZE>>>(
                A[aid], current_batch, ACCORD_SIZE);

        // ── copiar predições deste batch para CPU ──
        cudaMemcpy(result_gender, A[gid],
           current_batch * GENDER_SIZE * sizeof(float),
           cudaMemcpyDeviceToHost);

        cudaMemcpy(result_accord, A[aid],
           current_batch * ACCORD_SIZE * sizeof(float),
           cudaMemcpyDeviceToHost);

        // ── copiar predições para arrays acumulados (para ROC depois) ──
        for(int i = 0; i < current_batch; i++){
            for(int c = 0; c < GENDER_SIZE; c++)
                all_probs_gender[(b * batch_size + i) * GENDER_SIZE + c] = result_gender[i * GENDER_SIZE + c];
            for(int c = 0; c < ACCORD_SIZE; c++)
                all_probs_accord[(b * batch_size + i) * ACCORD_SIZE + c] = result_accord[i * ACCORD_SIZE + c];
        }

        // ── acumular matriz de confusão deste batch ──
        confusion_accumulate(conf_gender, result_gender, labels_gender + b * batch_size, current_batch, GENDER_SIZE);
        confusion_accumulate(conf_accord, result_accord, labels_accord + b * batch_size, current_batch, ACCORD_SIZE);

    }

    // ── calcular e imprimir todas as métricas ──
    metrics_used flags = {true, true, true, true, true};

    metrics_router(flags, conf_gender, all_probs_gender, labels_gender, test_data.num_rows, GENDER_SIZE, "outputs/roc_gender", verbosity);
    metrics_router(flags, conf_accord, all_probs_accord, labels_accord, test_data.num_rows, ACCORD_SIZE, "outputs/roc_accord", verbosity);

    // ── liberar memória ──
    confusion_free(conf_gender);
    confusion_free(conf_accord);

    for(int i = 0; i <= aid; i++)
        cudaFree(A[i]);

    delete[] A;
    delete[] result_gender;
    delete[] result_accord;
    delete[] all_probs_gender;
    delete[] all_probs_accord;
    delete[] labels_gender;
    delete[] labels_accord;
}


void NeuralNetwork::destroy_network(Network& net){

    for(int i = 0; i < net.n_shared; i++){
        _free_layer_(net.layers[i]);
    }
    _free_layer_(net.gender_head);
    _free_layer_(net.accord_head);

    delete[] net.layers;
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
