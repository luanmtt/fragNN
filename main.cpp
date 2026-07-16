#include <iostream>

#include "src/headers/dataloader.hpp"
#include "cuda/headers/nn.cuh"


typedef enum {
    ACT_SIGMOID     = 0,
    ACT_RELU        = 1,
    ACT_LEAKY_RELU  = 2,
    ACT_TANH        = 3
} ActivationType;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


int main(){
    
    cout << "Iniciando FragNN: \n";

    cout << "Testando load_csv: \n";
            
    Dataset train_t = load_csv(TRAIN_PATH);
    Dataset test_t = load_csv(TEST_PATH);


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    
    NeuralNetwork nn;
    
    int dims[] = {train_t.num_features, 128, 64};

    Network net = nn.create_network(dims, 2, NeuralNetwork::GENDER_SIZE, NeuralNetwork::ACCORD_SIZE, ACT_LEAKY_RELU, 0.001f, 0.9f);
    
    nn.train(net, train_t, 50, 32);
    nn.test(net, test_t, 32, 1);  // verbosity=1: per-class + AUC
    nn.write_weights(net, "weights.bin");
    nn.destroy_network(net);


    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


    cout << "Liberando memória. Fim do programa. \n";

    

    free_data(train_t);
    free_data(test_t);

    return 0;
}



// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
