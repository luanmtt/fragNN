#pragma once
#include <cuda_runtime.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// metrics — metrics.cu


struct metrics_used {
    bool PRECISION;
    bool RECALL;
    bool F1;
    bool ROC;
    bool ACCURACY;
};


struct Confusion {
    int* TP;
    int* FP;
    int* FN;
    int* TN;
    int total_correct;
    int total_samples;
};


void confusion_init(Confusion& C, int n_classes);

void confusion_free(Confusion& C);

void confusion_accumulate(  Confusion& C,
                            float* h_probs,
                            int* h_labels,
                            int current_batch,
                            int n_classes);

void precision(Confusion& C, float* out, int n_classes);

void recall(Confusion& C, float* out, int n_classes);

void f1(float* precision, float* recall, float* out, int n_classes);

float accuracy(Confusion& C);

float macro_avg(float* per_class, int n_classes);

void ROC(   float* probs,
            int* labels,
            int samples,
            int n_classes,
            const char* prefix,
            int verbosity);

void metrics_router(    metrics_used flags,
                        Confusion& C,
                        float* probs,
                        int* labels,
                        int samples,
                        int n_classes,
                        const char* prefix,
                        int verbosity);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
