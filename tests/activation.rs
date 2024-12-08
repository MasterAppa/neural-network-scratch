use neural_network_scratch::{matrix::{matrix::Matrix, MatrixTrait}, neural_network::{activation::Activation, layer::Layer}};


#[test]
fn test_linear_forward() {
    let mut linear_layer = Activation::Linear.to_layer();
    let input = Matrix::constant_matrix(2, 2, 1.0);
    let output = linear_layer.forward(input.clone());
    assert_eq!(output, input); // Linear activation should just return the input as output
}

// Test: Check forward pass for Sigmoid Activation Layer
#[test]
fn test_sigmoid_forward() {
    let mut sigmoid_layer = Activation::Sigmoid.to_layer();
    let input = Matrix::constant_matrix(2, 2, 0.0); // sigmoid(0) = 0.5
    let output = sigmoid_layer.forward(input.clone());
    let expected = Matrix::constant_matrix(2, 2, 0.5);
    assert_eq!(output, expected); // Sigmoid of 0 should be 0.5
}

// Test: Check backward pass for Linear Activation Layer
#[test]
fn test_linear_backward() {
    let mut linear_layer = Activation::Linear.to_layer();
    let output_gradient = Matrix::constant_matrix(2, 2, 2.0);
    linear_layer.forward(output_gradient.clone());
    
    let backward_output = linear_layer.backward(output_gradient);
    
    let expected = Matrix::constant_matrix(2, 2, 2.0); // Derivative of linear is 1, so grad remains the same
    
    assert_eq!(backward_output, expected);
}

// Test: Check backward pass for Sigmoid Activation Layer
#[test]
fn test_sigmoid_backward() {

    let mut sigmoid_layer = Activation::Sigmoid.to_layer();
    let output_gradient = Matrix::constant_matrix(2, 2, 2.0);

    // Forward pass (this will compute the sigmoid of the input)
    let input = Matrix::constant_matrix(2, 2, 0.0); //0 input, sigmoid(0) = 0.5
    sigmoid_layer.forward(input.clone());

    let backward_output = sigmoid_layer.backward(output_gradient.clone());

    // The expected value for sigmoid derivative at 0 should be 0.25 (sigmoid(0) = 0.5, derivative = 0.5 * 0.5)
    let expected = Matrix::constant_matrix(2, 2, 0.5); // Expected output is gradient * derivative (2.0 * 0.25)

    assert_eq!(backward_output, expected);
}
