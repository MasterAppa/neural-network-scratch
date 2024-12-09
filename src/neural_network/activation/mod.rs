use serde::{Deserialize, Serialize};

use crate::matrix::{matrix::Matrix, MatrixTrait};

use super::layer::Layer;


pub mod linear;
pub mod sigmoid;

pub type ActivationFn = fn(&Matrix) -> Matrix;
pub type DeriActivationFn = fn(&Matrix,&Matrix) -> Matrix;

pub enum ActivationFnDerivative{
    ActivationFn(ActivationFn),
    DeriActivationFn(DeriActivationFn)
}

pub struct ActivationLayer{
    input :  Option<Matrix>,
    output : Option<Matrix>,
    activation : ActivationFn,
    derivative : ActivationFnDerivative
}

impl ActivationLayer {
    pub fn new(activation: ActivationFn, derivative: ActivationFnDerivative) -> Self{
        Self {
            input: None,
            output: None,
            activation,
            derivative,
        }
    }
}

impl Layer for ActivationLayer{
    fn forward(&mut self, input: Matrix) -> Matrix {
        self.input = Some(input.clone());
        let output = (self.activation)(&input);
        self.output = Some(output.clone());
        output 
    }
    fn backward(&mut self, epoch: usize, output_gradient: Matrix) -> Matrix {
        match self.derivative{
            ActivationFnDerivative::ActivationFn(f) => {
                let input_temp = self.input.clone().unwrap();
                let derivative_result = (f)(&input_temp);
                output_gradient.matrix_mul(&derivative_result)
            }
            ActivationFnDerivative::DeriActivationFn(f) => {
                let output = self.output.clone().unwrap();
                (f)(&output,&output_gradient)
            }
        }
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