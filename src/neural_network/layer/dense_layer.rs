use ndarray::Axis;
use rand::distributions::weighted;

use crate::{matrix::{matrix::Matrix, Float, MatrixTrait}, neural_network::{initializer::Initializers, optimizer::Optimizers}};

use super::{Layer, LearnableLayer};

pub struct DenseLayer{
    input : Option<Matrix>,
    pub weights : Matrix,
    pub bias: Matrix,
    weights_optimizer: Optimizers,
    bias_optimizer: Optimizers
}

impl DenseLayer{
    fn new(nrows: usize, ncols: usize, weights_optimizer: Optimizers, bias_optimizer: Optimizers, weight_initializer : Initializers, bias_initializer: Initializers) -> Self{
        let weights = weight_initializer.gen_matrix(nrows, ncols);
        let bias = bias_initializer.gen_vector(nrows);

        Self{
            input : None,
            weights : weights,
            bias : bias,
            weights_optimizer : weights_optimizer,
            bias_optimizer : bias_optimizer
        }
    }
}

impl Layer for DenseLayer{
    fn forward(&mut self, input: Matrix) -> Matrix {

        //Considering weights: W, input: X, bias: b 
        // Considering Z the output with the shape (n,m)
        // The feed forward formula is z(t) =  w(t)*x
        let weighted_input = input.matrix_mul(&self.weights);
        let weighted_shape = weighted_input.shape();
        let res = weighted_input.component_add(&self.bias.broadcast(weighted_shape));

        
        self.input = Some(input);
        res
    }

    fn backward(&mut self, epoch: usize, output_gradient: Matrix) -> Matrix {
        // Ensure input is available or return an error (handle None safely)
        // Maybe should do this for an easier debugging
        let input = match &self.input {
            Some(input) => input,
            None => panic!("Input is None. This layer has not been forwarded yet."),
        };
    
        // Calculate the gradient with respect to weights (dL/dW): output_gradient * input.T
        let weights_gradient = output_gradient.dot(&input.transpose());
    
        // Calculate the gradient with respect to biases (dL/db): sum over the rows (examples in the batch)
        let biases_gradient = output_gradient.sum_axis(Axis(0));  // Sum along the rows (batch size)
    
        // Calculate the gradient with respect to the input (dL/dInput): self.weights.T * output_gradient
        let input_gradient = self.weights.transpose().dot(&output_gradient);
    
        // Update the weights using the optimizer
        self.weights = self.weights_optimizer.update( &self.weights, &weights_gradient,epoch);
    
        // Update the biases using the optimizer
        self.bias = self.bias_optimizer.update(&self.bias, &biases_gradient, epoch);
    
        // Return the input gradient for the next layer in the backpropagation
        input_gradient
    }
}


impl LearnableLayer for DenseLayer {
    
    fn get_learnable_parameters(&self) -> Vec<Vec<Float>> {
        let mut params = self.weights.get_data_col_leading();
        params.push(self.bias.get_column(0));
        params
    }

    
    fn set_learnable_parameters(&mut self, params_matrix: &Vec<Vec<Float>>) {
        let (weights, biases) = params_matrix.split_at(params_matrix.len() - 1); // Split weights and biases
        self.weights = Matrix::from_column_leading_vector2(&weights.to_vec());
        self.bias = Matrix::from_column_vector(&biases[0]);           
    }
}