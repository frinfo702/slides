use crate::tensor::Tensor2D;
use ndarray::{Array1, Array2};

pub struct Linear {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

impl Linear {
    pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        // 🔥 君の実装部分
        // flattenして (batch, features) x (features, out) + bias
        unimplemented!()
    }
}
