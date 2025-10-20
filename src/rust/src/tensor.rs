use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct Tensor2D {
    pub data: Array2<f32>,
}

impl Tensor2D {
    pub fn new(data: Array2<f32>) -> Self {
        Self { data }
    }

    pub fn shape(&self) -> (usize, usize) {
        // To get the shape, access self.data. Also, fix return type (usize, usize)
        self.data.dim()
    }
}
