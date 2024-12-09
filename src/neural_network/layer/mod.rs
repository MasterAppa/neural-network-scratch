

use crate::matrix::matrix::Matrix;
use crate::matrix::Float;
use crate::neural_network::activation::Activation;

pub mod dense_layer;
pub enum Layers{
    //Layers can be Dense Layers or just Activation Layers
    Dense,
    Activation(Activation),
}

pub trait Layer{
    // Layer must have a mechanism of forwarding to give the output
    // Layer must have a mechanism of backpropagation

    fn forward(&mut self, input: Matrix) -> Matrix;

    fn backward(&mut self, epoch: usize, output_gradient: Matrix) -> Matrix;
}


pub trait ParameterableLayer {
    fn as_learnable_layer(&self) -> Option<&dyn LearnableLayer>;
    fn as_learnable_layer_mut(&mut self) -> Option<&mut dyn LearnableLayer>;
    fn as_dropout_layer(&mut self) -> Option<&mut dyn DropoutLayer>;
}

pub trait DropoutLayer {
    fn enable_dropout(&mut self);
    fn disable_dropout(&mut self);
}

pub trait LearnableLayer {
    fn get_learnable_parameters(&self) -> Vec<Vec<Float>>;
    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Float>>);
}