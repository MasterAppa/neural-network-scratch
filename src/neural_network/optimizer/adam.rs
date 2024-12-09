use ndarray::Zip;
use serde::{Deserialize, Serialize};

use crate::{matrix::{matrix::Matrix, Float, MatrixTrait}, neural_network::learning_rate::{default_learning_rate, LearningRateSchedule}};


fn default_beta1() -> Float {
    0.9
}

fn default_beta2() -> Float {
    0.999
}

fn default_epsilon() -> Float {
    1e-8
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamOptimizer {
    // Hyperparameters
    #[serde(default = "default_learning_rate")]
    pub learning_rate_schedule: LearningRateSchedule,          // Learning rate
    #[serde(default = "default_beta1")]
    pub beta1: Float, /// Exponential decay rate for first moment estimate
    #[serde(default = "default_beta2")]
    pub beta2: Float, /// Exponential decay rate for second moment estimate
    #[serde(default = "default_epsilon")]
    pub epsilon: Float,      /// Small constant to prevent division by zero
    pub m: Option<Matrix>, // First moment vector
    pub v: Option<Matrix>, // Second moment vector
    pub t: usize,            // Time step
}

impl AdamOptimizer{
    pub fn new(learning_rate_schedule: LearningRateSchedule, beta1: Float, beta2: Float, epsilon: Float) -> Self {
        AdamOptimizer {
            learning_rate_schedule,
            beta1,
            beta2,
            epsilon,
            m: None,
            v: None,
            t: 0,
        }
    }

    pub fn init(&mut self, nrows: usize, ncols: usize) {
        let zeros = Matrix::zeros(nrows, ncols);
        self.m = Some(zeros.clone());
        self.v = Some(zeros);
    }

    pub fn default() -> Self {
        Self {
            v: None,
            m: None,
            beta1: default_beta1(),
            beta2: default_beta2(),
            learning_rate_schedule: default_learning_rate(),
            epsilon: default_epsilon(),
            t: 0,
        }
    }

    // Update the parameters using Adam's update rule
    pub fn update(&mut self, params: &Matrix, grads: &Matrix, epoch: usize) -> Matrix {
        // Increment time step
        self.t += 1;
        let (nrow, ncol) = grads.shape();
        if self.m.is_none() {
            self.m = Some(Matrix::zeros(nrow, ncol));
        }
        if self.v.is_none() {
            self.v = Some(Matrix::zeros(nrow, ncol));
        }
        // Safely unwrap moment vectors
        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first moment estimate (m)
        Zip::from(&mut m.0)
            .and(&grads.0)
            .for_each(|m_elem, &grad_elem| {
                *m_elem = self.beta1 * *m_elem + (1.0 - self.beta1) * grad_elem;
            });

        // Update biased second moment estimate (v)
        Zip::from(&mut v.0)
            .and(&grads.0)
            .for_each(|v_elem, &grad_elem| {
                *v_elem = self.beta2 * *v_elem + (1.0 - self.beta2) * grad_elem.powi(2);
            });

        // Bias correction
        let m_hat = m.0.mapv(|m_val| m_val / (1.0 - self.beta1.powi(self.t as i32)));
        let v_hat = v.0.mapv(|v_val| v_val / (1.0 - self.beta2.powi(self.t as i32)));

        // Get the current learning rate for this epoch
        let lr = self.learning_rate_schedule.get_learning_rate(epoch);
        let mut params_mut = params.clone();
        // Update parameters using Adam's rule
        Zip::from(&mut params_mut.0)
            .and(&m_hat)
            .and(&v_hat)
            .for_each(|param, &m_val, &v_val| {
                *param -= lr * m_val / (v_val.sqrt() + self.epsilon);
            });
        params_mut.clone()
    }

    pub fn update_parameters(
        &mut self,
        epoch: usize,
        parameters: &Matrix,
        parameters_gradient: &Matrix,
    ) -> Matrix {
        // Increment time step
        self.t += 1;

        // Get learning rate for current epoch
        let alpha = self.learning_rate_schedule.get_learning_rate(epoch);

        // Get dimensions of the gradient
        let (nrow, ncol) = parameters_gradient.shape();

        // Initialize moment vectors if not already initialized
        if self.m.is_none() {
            self.m = Some(Matrix::zeros(nrow, ncol));
        }
        if self.v.is_none() {
            self.v = Some(Matrix::zeros(nrow, ncol));
        }

        // Retrieve moment vectors
        let m = self.m.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();

        // Update first moment estimate (m)
        let m_updated = m
            .scalar_mul(self.beta1)
            .component_add(&parameters_gradient.scalar_mul(1.0 - self.beta1));

        // Update second moment estimate (v)
        let v_updated = v
            .scalar_mul(self.beta2)
            .component_add(&parameters_gradient.component_mul(parameters_gradient).scalar_mul(1.0 - self.beta2));

        // Bias correction
        let m_bias_corrected = m_updated.scalar_div(1.0 - self.beta1.powi(self.t as i32));
        let v_bias_corrected = v_updated.scalar_div(1.0 - self.beta2.powi(self.t as i32));

        // Square root of second moment estimate
        let v_sqrt = v_bias_corrected.sqrt();

        // Store updated moment vectors
        self.m = Some(m_updated.clone());
        self.v = Some(v_updated.clone());

        // Update parameters
        parameters.component_sub(
            &m_bias_corrected
                .scalar_mul(alpha)
                .component_div(&v_sqrt.scalar_add(self.epsilon))
        )
    }
}