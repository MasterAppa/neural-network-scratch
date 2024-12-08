use super::{ActivationFnDerivative, ActivationLayer};
use crate::matrix::MatrixTrait;

pub fn new() -> ActivationLayer{
    ActivationLayer::new(
        |m| m.sigmoid().clone(),
        ActivationFnDerivative::ActivationFn(|m| m.sigmoid_derivative().clone())
    )
}

