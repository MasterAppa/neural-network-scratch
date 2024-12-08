use ndarray::Zip;
use serde::{Deserialize, Serialize};

use crate::{matrix::{matrix::Matrix, Float, MatrixTrait}, neural_network::learning_rate::LearningRateSchedule};



fn default_beta1() -> Float{
    0.9
}
fn default_beta2() -> Float{
    0.999
}
fn default_alpha() -> Float{
    0.001
}
fn default_epsilon() -> Float{
    1e-8
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AdamOptimizer {
    // Hyperparameters
    pub alpha: LearningRateSchedule,          // Learning rate
    pub beta1: Float,          // Exponential decay rate for first moment estimate
    pub beta2: Float,          // Exponential decay rate for second moment estimate
    pub epsilon: Float,        // Small constant to prevent division by zero
    pub m: Option<Matrix>, // First moment vector
    pub v: Option<Matrix>, // Second moment vector
    pub t: usize,            // Time step
}

impl AdamOptimizer{
    pub fn new(alpha: LearningRateSchedule, beta1: Float, beta2: Float, epsilon: Float) -> Self {
        AdamOptimizer {
            alpha,
            beta1,
            beta2,
            epsilon,
            m: None,
            v: None,
            t: 0,
        }
    }
    pub fn init(&mut self, nrows:usize,ncols:usize) {
        let zeros = Matrix::zeros(nrows, ncols);
        self.m = Some(zeros.clone());
        self.v = Some(zeros);
    }

    // Update the parameters using Adam's update rule
    pub fn update(&mut self, params: &mut Matrix, grads: &Matrix, epoch: usize) {
        self.t += 1;

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first moment estimate (m)
        Zip::from(&mut m.0)  // Dereference Matrix to get Array2<Float>
            .and(&grads.0)    // Dereference Matrix to get Array2<Float>
            .apply(|m_elem, grad_elem| *m_elem = self.beta1 * *m_elem + (1.0 - self.beta1) * grad_elem);

        // Update biased second moment estimate (v)
        Zip::from(&mut v.0)  // Dereference Matrix to get Array2<Float>
            .and(&grads.0)    // Dereference Matrix to get Array2<Float>
            .apply(|v_elem, grad_elem| *v_elem = self.beta2 * *v_elem + (1.0 - self.beta2) * grad_elem.powi(2));

        // Bias correction
        let m_hat = m.0.mapv(|m_val| m_val / (1.0 - self.beta1.powi(self.t as i32)));
        let v_hat = v.0.mapv(|v_val| v_val / (1.0 - self.beta2.powi(self.t as i32)));

        // Get the current learning rate for this epoch
        let lr = self.learning_rate_schedule.get_learning_rate(epoch);

        // Update parameters using Adam's rule
        Zip::from(&mut params.0)   // Dereference Matrix to get Array2<Float>
            .and(&m_hat)            // Dereference Matrix to get Array2<Float>
            .and(&v_hat)            // Dereference Matrix to get Array2<Float>
            .apply(|param, m_val, v_val| {
                *param -= lr * m_val / (v_val.sqrt() + self.epsilon);
            });
    }
}