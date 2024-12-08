use adam::AdamOptimizer;

pub mod adam;

pub enum Optimizers {
    AdamOptimizer(AdamOptimizer),
}