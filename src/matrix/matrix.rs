
use ndarray::Array2;
use rand_distr::{Uniform, Normal, Distribution};
use rand::thread_rng;
use std::fmt;

use super::{Float, MatrixTrait, E};


#[derive(Clone, Debug, PartialEq)]
pub struct Matrix(pub Array2<Float>);


impl MatrixTrait for Matrix{

    fn zeros(nrows : usize, ncols: usize) -> Self {
        Self(Array2::zeros((nrows,ncols)))
    }

    fn identity(n: usize) -> Self {
        Self(Array2::eye(n))
    }

    fn constant_matrix(nrows : usize,ncols: usize,elem: Float) -> Self{
        Self(Array2::from_elem((nrows,ncols), elem))
    }

    fn ones(nrows : usize,ncols: usize) -> Self{
        Self(Array2::ones((nrows,ncols)))
    }

    
    fn shape(&self) -> (usize,usize) {
        (self.0.nrows(), self.0.ncols())
    }

    fn random_matrix_from_distribution( nrows: usize, ncols: usize, distribution: impl Distribution<Float>,) -> Self {
        let mut rng = thread_rng();
        let data: Vec<Float> = (0..(nrows * ncols))
            .map(|_| distribution.sample(&mut rng))
            .collect();
        Self(Array2::from_shape_vec((nrows, ncols), data).unwrap())
    }

    fn random_normal(nrows: usize, ncols: usize, mean: Float, std: Float) -> Self {
        let normal = Normal::new(mean, std).unwrap();
        Matrix::random_matrix_from_distribution(nrows, ncols, normal)
    }

    fn random_uniform(nrows: usize, ncols: usize, low: Float, high: Float) -> Self {
        let uniform = Uniform::new(low, high);
        Matrix::random_matrix_from_distribution(nrows, ncols, uniform)
    }

    fn xavier_uniform(nrows: usize, ncols: usize) -> Self {
        let limit = (6.0 / (nrows as Float + ncols as Float)).sqrt();
        let uniform = Uniform::new(-limit, limit);
        Matrix::random_matrix_from_distribution(nrows, ncols, uniform)
    }

    fn xavier_normal(nrows: usize, ncols: usize) -> Self {
        let std_dev = (2.0 / (nrows as Float + ncols as Float)).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        Matrix::random_matrix_from_distribution(nrows, ncols, normal)
    }

    fn float_mul(&self, value: Float) -> Self {
        Self(&self.0 * value)
    }

    
    fn exp(&self) -> Self {
        Self(self.0.mapv(Float::exp))
    }

    fn sigmoid(&self) -> Self {
        let sigmoid_matrix = self.0.mapv(|x| 1.0 / (1.0 + E.powf(-x)));
        Self(sigmoid_matrix)
    }

    fn sigmoid_derivative(&self) -> Self {
        // Compute sigmoid and its derivative
        let sigmoid_matrix = self.sigmoid();
        let result_matrix = sigmoid_matrix.0.mapv(|x| x * (1.0 - x));  // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        Self(result_matrix)
    }

    fn matrix_mul(&self, other: &Matrix) -> Self {
        Self(&self.0 * &other.0)
    }

    fn dot(&self, other: &Matrix) -> Self{
        let result = self.0.dot(&other.0);
        Self(result.clone())
    }
}


impl Matrix {
    pub fn print(&self) {
        println!("{:?}", self.0);
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix {}x{}:\n", self.0.nrows(), self.0.ncols())?;
        for i in 0..self.0.nrows() {
            for j in 0..self.0.ncols() {
                write!(f, "{}\t", self.0[[i, j]])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

