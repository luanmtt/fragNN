#pragma once
#include <cuda_runtime.h>

#define EPSILON 1e-15f

__device__ float cross_entropy(float* probs, int label, int n_classes);

__device__ float mse(float* pred, float* target, int n_elements);

__device__ float focal_loss(float* probs, int label, int n_classes, float gamma);

__device__ float huber(float pred, float target, float delta);
