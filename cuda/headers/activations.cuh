#pragma once
#include <cuda_runtime.h>


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__device__ float sigmoid(float x);
__device__ float sigmoid_backp(float x_activated);

__device__ float leaky_relu(float x, float alpha);
__device__ float leaky_relu_backp(float x_activated, float alpha);

__device__ float relu(float x, float alpha);
__device__ float relu_backp(float x_activated, float alpha);

__device__ float tanh_(float x);
__device__ float tanh_backp(float x_activated);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
