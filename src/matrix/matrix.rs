
use ndarray::Array2;
use rand_distr::{Normal, Distribution};
use rand::thread_rng;
use std::fmt;

use super::{Float, MatrixTrait, E};


#[derive(Clone, Debug)]
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

    fn random_normal(nrows : usize,ncols: usize, mean: Float, std: Float) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(mean, std).unwrap();

        let data: Vec<Float> = (0..(nrows*ncols)).map(|_| normal.sample(&mut rng)).collect();

        Self(Array2::from_shape_vec((nrows,ncols), data).unwrap())
    }

    
    fn float_mul(&self, value: Float) -> Self {
        Self(&self.0 * value)
    }

    
    fn exp(&self) -> Self {
        Self(self.0.mapv(Float::exp))
    }
    
    // Apply the sigmoid function to each element in the matrix
    fn sigmoid(&self) -> Self {
        let sigmoid_matrix = self.0.mapv(|x| 1.0 / (1.0 + E.powf(-x)));
        Self(sigmoid_matrix)
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

