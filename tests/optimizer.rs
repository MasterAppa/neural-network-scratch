use neural_network_scratch::{matrix::{matrix::Matrix, MatrixTrait}, neural_network::{learning_rate::LearningRateSchedule, optimizer::adam::AdamOptimizer}};




#[test]
fn test_adam_optimizer() {
    let learning_rate_schedule = LearningRateSchedule::Constant(0.001);
    let mut adam = AdamOptimizer::new(learning_rate_schedule, 0.9, 0.999, 1e-8);

    let mut params = Matrix::random_normal(3, 3, 0.0, 0.1);
    let grads = Matrix::random_normal(3, 3, 0.0, 0.05);

    adam.init(params.shape().0,params.shape().1);

    for epoch in 1..=5 {
        adam.update(&mut params, &grads, epoch);
        println!("Epoch {}: {:?}", epoch, params);
    }
}