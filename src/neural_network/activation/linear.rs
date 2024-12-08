use crate::matrix::{matrix::Matrix, MatrixTrait};

use super::{ActivationFnDerivative, ActivationLayer};

pub fn new() -> ActivationLayer{
    ActivationLayer::new(
        |m| m.clone(),
        ActivationFnDerivative::ActivationFn(|m: &Matrix| Matrix::ones(m.shape().0, m.shape().1))
    )
}