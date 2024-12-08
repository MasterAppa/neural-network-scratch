use serde::{Deserialize, Serialize};

use crate::matrix::Float;

pub fn default_learning_rate() -> LearningRateSchedule {
    LearningRateSchedule::Constant(0.001)
}

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