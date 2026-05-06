#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


using namespace std;


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// struct Dataset & paths


struct Dataset {
    
    float* X;
    int* y_gender;
    int* y_accord;
    int num_features;
    int num_rows;

};

constexpr const char* TRAIN_PATH = "data/train.csv";
constexpr const char* TEST_PATH = "data/test.csv";


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  declaração de funções em dataloader.cpp


Dataset load_csv(const string &path);


void free_data(Dataset& data_t);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
