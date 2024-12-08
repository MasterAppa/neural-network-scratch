use super::ActivationLayer;
use crate::matrix::MatrixTrait;

pub fn new() -> ActivationLayer{
    ActivationLayer::new(
        |m| m.sigmoid().clone()
    )
}

