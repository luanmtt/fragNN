#include "kernels.cuh"
#include "headers/loss.cuh"


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

constexpr float EPSILON = 1e-15f;


__device__ float mse(float* pred, float* target, int n){

    /* 
        cost = (1/n) * Σ(y_true - y_pred)² 

        mean squared error: penalizes large errors quadratically

    */
    
    float cost = 0.0f;

    for(int i = 0; i < n; i++){
        
        float temp = (pred[i] - target[i]);
        cost += temp * temp; 
    }

    return cost / (float)n;
    
}


__device__ float cross_entropy(float* probabilities, int label, int n_classes){

    /* 
        
        cross entropy: measures distance between predicted probabilities and true label 
        -log(probs[label]) — punishes confident wrong predictions heavily

    */

    const float correct_prob = probabilities[label];
    return -__logf(correct_prob + EPSILON);

}



__device__ float focal(float* probabilities, int label, int n_classes, float gamma){
        
    /* 
        cost = (1 - correct_prob)^γ * log(correct_prob + ε)

        focal loss: down-weights easy examples, focuses on hard ones
        quando correct_prob é alto, ou seja, muita certeza, o erro
        custo perto de 0. quando correct_prob é baixo, pouca certeza,
        custo perto de 1.

    */

    const float correct_prob = probabilities[label];
    const float focus =  __powf(-(1.0 - correct_prob), gamma);

    return -focus * __logf(correct_prob + EPSILON);

}


__device__ float huber(float pred, float target, float delta){

    /* 

        quadratic when |error| <= delta, linear beyond — smoother than mse on outliers

    */

    float diff = fabsf(pred - target);
        
    if(diff <= delta)
        return 0.5f * diff * diff;
    
    else 
        return delta * (diff - 0.5 * delta);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
