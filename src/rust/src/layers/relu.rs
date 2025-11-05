use crate::tensor::Tensor2D;

pub struct ReLU;

impl ReLU {
    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        Tensor2D::new(input.data.mapv(|val| val.max(0.0)))
    }
}
