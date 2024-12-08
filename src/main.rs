use matrix::{Matrix, MatrixTrait};

mod matrix;



fn main() {
    let i = Matrix::random_normal(3, 3, 5.0, 2.1);
    println!("{i}");
    i.print();
}
