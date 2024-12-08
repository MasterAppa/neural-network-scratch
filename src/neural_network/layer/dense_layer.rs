use crate::matrix::matrix::Matrix;

pub struct DenseLayer{
    input : Option<Matrix>,
    weights : Matrix,
    bias: Matrix
}