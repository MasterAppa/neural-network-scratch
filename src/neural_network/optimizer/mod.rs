use adam::AdamOptimizer;

use crate::matrix::matrix::Matrix;

pub mod adam;

pub enum Optimizers {
    AdamOptimizer(AdamOptimizer),
}


impl Optimizers {
    pub fn update(
        &mut self,
        parameters: &Matrix,
        parameters_gradient: &Matrix,
        epoch: usize
    ) -> Matrix {
        match self {
            Optimizers::AdamOptimizer(adam) => {
                adam.update(parameters, parameters_gradient,epoch)
            }
        }
    }
}