use ndarray::Array2;
use neural_network_scratch::matrix::{Matrix, MatrixTrait};


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