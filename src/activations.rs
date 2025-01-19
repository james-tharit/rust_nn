use ndarray::prelude::*;
use std::f32::consts::E;

// Activation functions introduce non-linearity to neural networks and play a crucial role in the forward propagation process.
// The code provides implementations for two commonly used activation functions: sigmoid and relu.

#[derive(Debug)]
pub struct ActivationCache {
    pub z: Array2<f32>,
}

#[derive(Debug)]
pub struct LinearCache {
    pub a: Array2<f32>,
    pub w: Array2<f32>,
    pub b: Array2<f32>,
}

//The sigmoid function takes a single value z as input and returns the sigmoid activation, which is calculated using the sigmoid formula: 1 / (1 + e^-z).
pub fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

// The relu function takes a single value z as input and applies the Rectified Linear Unit (ReLU) activation. If z is greater than zero, the function returns z; otherwise, it returns zero.
pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| sigmoid(&x)), ActivationCache { z })
}

pub fn relu_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| relu(&x)), ActivationCache { z })
}

//The linear_forward function takes the activation matrix a, weight matrix w, and bias matrix b as inputs.
//It performs the linear transformation by calculating the dot product of w and a, and then adding b to the result.
//The resulting matrix z represents the logits of the layer. The function returns z along with a LinearCache struct that stores the input matrices for later use in backward propagation.
pub fn linear_forward(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };
    return (z, cache);
}

pub fn linear_forward_activation(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    activation: &str,
) -> Result<(Array2<f32>, (LinearCache, ActivationCache)), String> {
    match activation {
        "sigmoid" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = sigmoid_activation(z);
            return Ok((a_next, (linear_cache, activation_cache)));
        }
        "relu" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = relu_activation(z);
            return Ok((a_next, (linear_cache, activation_cache)));
        }
        _ => return Err("wrong activation string".to_string()),
    }
}
