use neural_network_scratch::matrix::{matrix::Matrix, MatrixTrait};




fn main() {
    let i = Matrix::random_normal(3, 3, 5.0, 2.1);
    println!("{i}");
    i.print();
}
