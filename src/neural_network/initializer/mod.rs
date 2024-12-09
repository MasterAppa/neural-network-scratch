use serde::{Deserialize, Serialize};

use crate::matrix::{matrix::Matrix, MatrixTrait};


#[derive(Serialize, Debug, Deserialize, Clone)]
pub enum Initializers {
    Zeros,
    Uniform,
    UniformSigned,
    Normal,
    XavierUniform,
    XavierNormal
}

impl Initializers {
    fn gen_matrix_internal(&self, nrows: usize, ncols: usize) -> Matrix {
        match self {
            Initializers::Zeros => Matrix::zeros(nrows, ncols),
            Initializers::Uniform => Matrix::random_uniform(nrows, ncols, 0.0, 1.0),
            Initializers::UniformSigned => Matrix::random_uniform(nrows, ncols, -1.0, 1.0),
            Initializers::Normal => Matrix::random_normal(nrows, ncols, 0.0, 1.0),
            Initializers::XavierUniform => Matrix::xavier_uniform(nrows, ncols),
            Initializers::XavierNormal => Matrix::xavier_normal(nrows, ncols),
        }
    }
    // Generates a matrix of size (nrows, 1), a column vector
    pub fn gen_vector(&self, nrows: usize) -> Matrix {
        self.gen_matrix_internal(nrows, 1)
    }

    // Generates a matrix of size (nrows, ncols)
    pub fn gen_matrix(&self, nrows: usize, ncols: usize) -> Matrix {
        self.gen_matrix_internal(nrows, ncols)
    }
}