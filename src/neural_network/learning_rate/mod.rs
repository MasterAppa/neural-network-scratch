use serde::{Deserialize, Serialize};

use crate::matrix::Float;



#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant(Float)
}

impl LearningRateSchedule {
    pub fn get_learning_rate(&self, epoch: usize) -> Float {
        match self {
            LearningRateSchedule::Constant(c) => *c,
        }
    }

    
}