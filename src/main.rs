use neural_network_scratch::{matrix::{matrix::Matrix, MatrixTrait}, neural_network::{learning_rate::LearningRateSchedule, optimizer::adam::AdamOptimizer}};




fn test_adam_optimizer(params : &Matrix, grads: &Matrix) {
    let learning_rate_schedule = LearningRateSchedule::Constant(0.001);
    let mut adam = AdamOptimizer::new(learning_rate_schedule, 0.9, 0.999, 1e-8);

    let mut params_mut = params.clone();
    let grads = grads.clone();

    adam.init(params_mut.shape().0,params_mut.shape().1);

    for epoch in 1..=5 {
        params_mut = adam.update(&params_mut, &grads,epoch);
        println!("Epoch {}: {:?}", epoch, params_mut);
    }
}

fn test_adam_optimizer2(params : &Matrix, grads: &Matrix) {
    let learning_rate_schedule = LearningRateSchedule::Constant(0.001);
    let mut adam = AdamOptimizer::new(learning_rate_schedule, 0.9, 0.999, 1e-8);

    let mut params_mut = params.clone();
    let grads = grads.clone();

    adam.init(params_mut.shape().0,params_mut.shape().1);

    for epoch in 1..=5 {
        params_mut = adam.update_parameters(epoch,&params_mut, &grads);
        println!("Epoch {}: {:?}", epoch, params_mut);
    }
}

fn main() {
    let params = Matrix::random_normal(3, 3, 0.0, 0.1);
    let grads = Matrix::random_normal(3, 3, 0.0, 0.05);
    test_adam_optimizer(&params,&grads);
    println!("Second approach");
    test_adam_optimizer2(&params,&grads);
}
