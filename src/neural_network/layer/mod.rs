

use crate::matrix::matrix::Matrix;
use crate::neural_network::activation::Activation;

pub enum Layers{
    //Layers can be Dense Layers or just Activation Layers
    Dense,
    Activation(Activation),
}

pub trait Layer{
    // Layer must have a mechanism of forwarding to give the output
    // Layer must have a mechanism of backpropagation

    fn forward(&mut self, input: Matrix) -> Matrix;

    fn backward(&mut self, output_gradient: Matrix) -> Matrix;
}