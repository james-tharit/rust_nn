use rust_nn::{layer::Layer, nueron::Nueron};

fn main() {
    let inputs: Vec<f32> = vec![1.0, 2.0, 3.0, 2.5];

    let n1 = Nueron::new(vec![0.2, 0.8, -0.5, 1.0], 2.0);
    let n2 = Nueron::new(vec![0.5, -0.91, 0.26, -0.5], 3.0);
    let n3 = Nueron::new(vec![-0.26, -0.27, 0.17, 0.87], 0.5);

    let layer = Layer::new(vec![n1, n2, n3]);

    let output: Vec<f32> = layer.output(&inputs);

    println!("{:?}", output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_neuron_layer_output() {
        let inputs: Vec<f32> = vec![1.0, 2.0, 3.0, 2.5];
        let n1 = Nueron::new(vec![0.2, 0.8, -0.5, 1.0], 2.0);
        let layer = Layer::new(vec![n1]);

        let output: Vec<f32> = layer.output(&inputs);

        // Expected output based on the calculations
        /*
            1*0.2 = .2
            2*0.8 = 1.6
            3*-0.5 = -1.5
            2.5*1 = 2.5
            sum + bias = 2.8+2 = 4
        */
        let expected_output: Vec<f32> = vec![4.8];

        assert_eq!(output, expected_output);
    }
}
