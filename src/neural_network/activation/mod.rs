use serde::{Deserialize, Serialize};

use crate::matrix::matrix::Matrix;

use super::layer::Layer;


pub mod linear;
pub mod sigmoid;

pub type ActivationFn = fn(&Matrix) -> Matrix;


pub struct ActivationLayer{
    input :  Option<Matrix>,
    output : Option<Matrix>,
    activation : ActivationFn
}

impl ActivationLayer {
    pub fn new(activation: ActivationFn) -> Self{
        Self { input: None, output: None, activation: activation }
    }
}

impl Layer for ActivationLayer{
    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = Some(input.clone());
        let output = (self.activation)(&input);
        self.output = Some(output.clone());
        output 
    }
    fn backward(&mut self, output_gradient: Matrix) -> Matrix {
        
    }
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation{
    Linear,
    Sigmoid
}

impl Activation{
    pub fn to_layer(&self) -> ActivationLayer{
        match self {
            Self::Linear => linear::new(),
            Self::Sigmoid => sigmoid::new()
        }
    }
}