use ndarray::Array2;
use neural_network_scratch::matrix::{matrix::Matrix, MatrixTrait, E};


#[test]
fn test_zeros(){
    let m_zeros = Matrix::zeros(3, 3);
    let expected = Array2::zeros((3,3));

    assert_eq!(m_zeros.0,expected)
}


#[test]
fn test_zeros_empty_matrix() {
    let matrix = Matrix::zeros(0, 0);

    let expected = Array2::zeros((0, 0));

    assert_eq!(matrix.0, expected);
}

#[test]
fn test_eye_matrix_3x3() {
    // Create a 3x3 identity matrix
    let matrix = Matrix::identity(3);
    let expected = Array2::from_shape_vec(
        (3, 3),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ).unwrap();
    assert_eq!(matrix.0, expected);
}

#[test]
fn test_eye_matrix_0x0() {
    // Create a 0x0 identity matrix (edge case)
    let matrix = Matrix::identity(0);
    let expected = Array2::zeros((0, 0));
    assert_eq!(matrix.0, expected);
}

// Test: Check shape of a matrix
#[test]
fn test_shape() {
    let m = Matrix::zeros(3, 4);
    assert_eq!(m.shape(), (3, 4));
}

// Test: Check random normal matrix generation
#[test]
fn test_random_normal() {
    let m = Matrix::random_normal(2, 3, 0.0, 1.0);
    assert_eq!(m.shape(), (2, 3));
    // Check if values are approximately normal
    assert!(m.0.iter().all(|&x| x >= -5.0 && x <= 5.0)); // Assuming values are within range
}

// Test: Check matrix multiplication (element-wise multiplication)
#[test]
fn test_float_mul() {
    let m = Matrix::constant_matrix(2, 2, 2.0);
    let result = m.float_mul(3.0);
    let expected = Matrix::constant_matrix(2, 2, 6.0);
    assert_eq!(result.0, expected.0);
}

// Test: Check exponentiation
#[test]
fn test_exp() {
    let m = Matrix::constant_matrix(2, 2, 1.0);
    let result = m.exp();
    let expected = Matrix::constant_matrix(2, 2, E.powf(1.0));
    assert_eq!(result.0, expected.0);
}

// Test: Check sigmoid function
#[test]
fn test_sigmoid() {
    let m = Matrix::constant_matrix(2, 2, 0.0);
    let result = m.sigmoid();
    let expected = Matrix::constant_matrix(2, 2, 0.5);  // sigmoid(0) = 0.5
    assert_eq!(result.0, expected.0);
}

// Test: Check sigmoid derivative
#[test]
fn test_sigmoid_derivative() {
    let m = Matrix::constant_matrix(2, 2, 0.0);
    let result = m.sigmoid_derivative();
    let expected = Matrix::constant_matrix(2, 2, 0.25);  // sigmoid'(0) = 0.25
    assert_eq!(result.0, expected.0);
}

// Test: Check matrix multiplication (dot product)
#[test]
fn test_matrix_mul() {
    let m1 = Matrix::constant_matrix(2, 3, 1.0);
    let m2 = Matrix::constant_matrix(3, 2, 2.0);
    let result = m1.matrix_mul(&m2);
    let expected = Matrix::constant_matrix(2, 2, 6.0); // 1*2 + 1*2 + 1*2 = 6
    assert_eq!(result.0, expected.0);
}

// Test: Check dot product
#[test]
fn test_dot() {
    let m1 = Matrix::constant_matrix(2, 3, 1.0);
    let m2 = Matrix::constant_matrix(3, 2, 2.0);
    let result = m1.dot(&m2);
    let expected = Matrix::constant_matrix(2, 2, 6.0); // Same logic as matrix multiplication
    assert_eq!(result.0, expected.0);
}

// Test: Check Matrix printing
#[test]
fn test_print() {
    let m = Matrix::constant_matrix(2, 2, 1.0);
    m.print();  // We can't assert on print output, but we can check that it compiles without errors
}

// Test: Display format of Matrix
#[test]
fn test_display() {
    let m = Matrix::constant_matrix(2, 2, 1.0);
    let display_str = format!("{}", m);
    assert!(display_str.contains("Matrix 2x2:"));
    assert!(display_str.contains("1.0\t1.0"));
}