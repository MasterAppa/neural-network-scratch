
use ndarray::Array2;
use polars::prelude::Scalar;
use rand_distr::{Normal, Distribution};
use rand::thread_rng;
use std::fmt;

#[cfg(feature = "f64")]
pub type Float = Float;

#[cfg(not(feature = "f64"))]
pub type Float = f32;

pub trait MatrixTrait: Clone {
    fn zeros(nrows : usize, ncols: usize) -> Self;
    fn identity(n: usize) -> Self;
    fn shape(&self) -> (usize,usize);
    fn random_normal(nrows : usize,ncols: usize, mean: Float, std: Float) -> Self;
}
#[derive(Clone, Debug)]
pub struct Matrix(pub Array2<Float>);


impl MatrixTrait for Matrix{
    fn zeros(nrows : usize, ncols: usize) -> Self {
        Self(Array2::zeros((nrows,ncols)))
    }
    fn identity(n: usize) -> Self {
        Self(Array2::eye(n))
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



