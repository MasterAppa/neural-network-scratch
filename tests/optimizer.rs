use neural_network_scratch::{matrix::{matrix::Matrix, MatrixTrait}, neural_network::learning_rate::LearningRateSchedule};




#[test]
fn test_adam_optimizer() {
    let learning_rate_schedule = LearningRateSchedule::Constant(0.001); // Use your constant learning rate
    let mut adam = AdamOptimizer::new(learning_rate_schedule, 0.9, 0.999, 1e-8);

    // Initialize parameters (weights) and gradients
    let mut params = Matrix::random_normal(3, 3, 0.0, 0.1);
    let grads = Matrix::random_normal(3, 3, 0.0, 0.05);

    // Initialize Adam optimizer with the shape of the parameters
    adam.init(params.shape());

    // Perform updates for 5 epochs
    for epoch in 1..=5 {
        adam.update(&mut params, &grads, epoch);
        println!("Epoch {}: {:?}", epoch, params);
    }
}