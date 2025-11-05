use crate::tensor::Tensor2D;

pub struct MaxPool2D {
    pub size: usize,
}

impl MaxPool2D {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        // 🔥 君の実装部分
        // stride=sizeのmax pooling
        unimplemented!()
    }
}
