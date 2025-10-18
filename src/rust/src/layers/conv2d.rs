use crate::tensor::Tensor2D;
use ndarray::{s, Array2};

pub struct Conv2D {
    pub weight: Array2<f32>,
    pub bias: f32,
}

impl Conv2D {
    pub fn new(weight: Array2<f32>, bias: f32) -> Self {
        Self { weight, bias }
    }

    /// Forward convolution (valid)
    pub fn forward(&self, input: &Tensor2D) -> Tensor2D {
        let (input_width, input_height) = input.data.dim();
        let (kernel_width, kernel_height) = self.weight.dim();
        let (output_width, output_height) = (
            input_width - kernel_width + 1,
            input_height - kernel_height + 1,
        );
        let mut output = Array2::<f32>::zeros((output_height, output_width));

        for i in 0..output_width {
            for j in 0..output_height {
                let region = input
                    .data
                    .slice(s![i..i + kernel_width, j..j + kernel_height]);
                let val = (&region * &self.weight).sum() + self.bias;
                output[[i, j]] = val;
            }
        }

        Tensor2D::new(output)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::Tensor2D;
    use ndarray::array;

    #[test]
    fn test_conv2d_forward() {
        let input = Tensor2D::new(array![
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [7.0, 8.0, 9.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);

        let kernel = array![[1.0, 0.0], [0.0, -1.0]];

        let conv = Conv2D::new(kernel, 0.5);
        let output = conv.forward(&input);

        let expected = array![[-3.5, -3.5, 3.5], [-3.5, -3.5, 6.5], [7.5, 8.5, 9.5]];

        let diff = (&output.data - &expected).mapv(f32::abs);
        let mean_err = diff.mean().unwrap();
        assert!(
            mean_err < 1e-4,
            "mean abs error too large: {}\n output: {:?}",
            mean_err,
            output
        );
    }
}
