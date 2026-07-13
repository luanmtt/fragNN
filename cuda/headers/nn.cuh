#pragma once

#include "../src/headers/dataloader.hpp"
#include "metrics.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// wrappers para __global__ kernels (chamáveis de main.cpp)


void launch_loss_kernel(    float* probs,
                            int* labels,
                            float* dl_dy,
                            int batch,
                            int n_classes);


void launch_sum_gradients(  float* dst,
                            float* src,
                            int n);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


struct Layer {

    int in_dim; 
    int out_dim;
    int activation_type;
    
    float* W;
    float* b;

    float* dW;
    float* db;

    float* m_W;
    float* v_W;

    float* m_b;
    float* v_b;
   
    float* mask;

};


struct Network {
    
    Layer* layers;
    int n_shared;

    Layer gender_head;
    Layer accord_head;

    float lr;

    float beta_1;
    float beta_2;
    float epsilon;          
    float keep_prob;
    int timestamp;

    int num_features;

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NeuralNetwork {

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
                    int batch_size);

        void write_weights(Network& net, const char* path);

        void load_weights(Network& net, const char* path);

};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
