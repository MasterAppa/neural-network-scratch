use crate::{matrix::{matrix::Matrix, Float, MatrixTrait}, neural_network::activation::ActivationLayer};
use super::{dense_layer::DenseLayer, DropoutLayer, Layer, LearnableLayer, ParameterableLayer};




#[derive(Debug)]
pub struct FullLayer {
    dense: DenseLayer,
    activation: ActivationLayer,
    dropout_enabled: bool,
    dropout_rate: Option<f64>,
    mask: Option<Matrix>,
}

impl FullLayer {
    pub fn new(
        dense: DenseLayer, 
        activation: ActivationLayer, 
        dropout_rate: Option<f64>
    ) -> Self {
        // Validate dropout rate if provided
        if let Some(rate) = dropout_rate {
            assert!(rate >= 0.0 && rate <= 1.0, "Dropout rate must be between 0 and 1");
        }

        Self {
            dense,
            activation,
            dropout_enabled: dropout_rate.is_some(),
            dropout_rate,
            mask: None,
        }
    }

    fn generate_dropout_mask(&mut self, output_shape: (usize, usize)) -> Matrix {
        let dropout_rate = self.dropout_rate.unwrap_or(0.0);
        
        // Generate a uniform random matrix
        let random_matrix = Matrix::random_uniform(
            output_shape.0, 
            output_shape.1, 
            0.0,  // mean
            1.0   // std
        );
        
        // Create a mask based on the dropout condition
        random_matrix.map(|&x| {
            if x < (1.0 - dropout_rate) {
                1.0 / (1.0 - dropout_rate)  // Scale to maintain expected value
            } else {
                0.0
            }
        })
    }
}

impl Layer for FullLayer {
    fn forward(&mut self, input: Matrix) -> Matrix {
        let dense_output = self.dense.forward(input);
        
        let output = if self.dropout_enabled {
            let mask = self.generate_dropout_mask(dense_output.shape());
            self.mask = Some(mask.clone());
            dense_output.component_mul(&mask)
        } else {
            dense_output
        };

        self.activation.forward(output)
    }

    fn backward(&mut self, epoch: usize, output_gradient: Matrix) -> Matrix {
        let activation_input_gradient = self.activation.backward(epoch, output_gradient);
        
        let input_gradient = match (self.dropout_enabled, &self.mask) {
            (true, Some(mask)) => {
                // Apply dropout mask to the gradient
                let dropout_gradient = activation_input_gradient.component_mul(mask);
                self.dense.backward(epoch, dropout_gradient)
            },
            _ => {
                // No dropout or mask not available
                self.dense.backward(epoch, activation_input_gradient)
            }
        };

        input_gradient
    }
}


impl ParameterableLayer for FullLayer {
    fn as_learnable_layer(&self) -> Option<&dyn LearnableLayer> {
        Some(self)
    }

    fn as_learnable_layer_mut(&mut self) -> Option<&mut dyn LearnableLayer> {
        Some(self)
    }

    fn as_dropout_layer(&mut self) -> Option<&mut dyn DropoutLayer> {
        Some(self)
    }
}

impl LearnableLayer for FullLayer {
    // returns a matrix of the (jxi) weights and the final column being the (j) biases
    fn get_learnable_parameters(&self) -> Vec<Vec<Float>> {
        self.dense.get_learnable_parameters()
    }

    // takes a matrix of the (jxi) weights and the final column being the (j) biases
    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Float>>) {
        self.dense.set_learnable_parameters(params_matrix)
    }
}

impl DropoutLayer for FullLayer {
    fn enable_dropout(&mut self) {
        self.dropout_enabled = true;
    }

    fn disable_dropout(&mut self) {
        self.dropout_enabled = false;
    }
}