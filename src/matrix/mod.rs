use rand_distr::Distribution;
use matrix::Matrix;

pub mod matrix;


pub type Float = f64;


pub const E: f64 = std::f64::consts::E;

pub trait MatrixTrait: Clone {

    ///Creates a Matrix with filled with 0s
    fn zeros(nrows : usize, ncols: usize) -> Self;

    ///Creates a Matrix with only the diagonal with 1s, identity matrix
    fn identity(n: usize) -> Self;

    ///Creates a Matrix where all the values have the value elem
    fn constant_matrix(nrows : usize,ncols: usize,elem: Float) -> Self;

    /// Creates a Matrix with random values that follow a distribution user defined
    fn random_matrix_from_distribution( nrows: usize, ncols: usize, distribution: impl Distribution<Float>,) -> Self;

    /// Creates a Matrix with random values that follow a normal distribution
    fn random_normal(nrows : usize,ncols: usize, mean: Float, std: Float) -> Self;
    /// Creates a Matrix with random values that follow a uniform distribution
    fn random_uniform(nrows: usize, ncols: usize, mean: Float, std: Float) -> Self;
    // Xavier/Glorot initialization using normal distribution
    fn xavier_normal(nrows: usize, ncols: usize) -> Self;
    // Xavier/Glorot initialization using uniform distribution
    fn xavier_uniform(n_in: usize, n_out: usize) -> Self;
    ///Returns the shape of the Matrix
    fn shape(&self) -> (usize,usize);

    ///Creates a Matrix where all the values are 1s
    fn ones(nrows : usize,ncols: usize) -> Self;

    ///Calculates the multiplication between a Matrix and a value
    fn float_mul(&self ,value: Float) -> Self;

    fn matrix_mul(&self, other: &Matrix) -> Self;

    fn dot(&self, other: &Matrix) -> Self;

    ///The Matrix returned will be a Matrix of the exponent e^(self[i,j])
    fn exp(&self) -> Self;

    fn sigmoid(&self) -> Self;

    fn sigmoid_derivative(&self) -> Self;

}

