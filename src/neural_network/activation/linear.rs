use super::ActivationLayer;

pub fn new() -> ActivationLayer{
    ActivationLayer::new(
        |m| m.clone()
    )
}