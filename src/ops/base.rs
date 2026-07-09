//! Base tensor operations for MLX backend.

use crate::tensor::MlxTensor;
use mlx_rs::ops::indexing::take_axis;
use mlx_rs::Array;

/// Base operations on f32 MLX tensors.
impl MlxTensor<f32> {
    /// Reshape this tensor.
    pub fn reshape_to(&self, shape: &[i32]) -> Self {
        let array = self.array.reshape(shape).expect("Failed to reshape array");
        MlxTensor::new(array, self.device)
    }

    /// Transpose this tensor (reverses all axes).
    pub fn transpose_all(&self) -> Self {
        let array = mlx_rs::ops::transpose(&self.array).expect("Failed to transpose array");
        MlxTensor::new(array, self.device)
    }

    /// Expand tensor to a new shape.
    pub fn broadcast_to(&self, shape: &[i32]) -> Self {
        let array =
            mlx_rs::ops::broadcast_to(&self.array, shape).expect("Failed to broadcast array");
        MlxTensor::new(array, self.device)
    }
}

/// Concatenate tensors along a dimension.
pub fn concat(tensors: &[&MlxTensor<f32>], dim: usize) -> MlxTensor<f32> {
    if tensors.is_empty() {
        panic!("Cannot concatenate empty list of tensors");
    }

    let device = tensors[0].device;
    let arrays: Vec<&mlx_rs::Array> = tensors.iter().map(|t| &t.array).collect();

    let array =
        mlx_rs::ops::concatenate_axis(&arrays, dim as i32).expect("Failed to concatenate arrays");
    MlxTensor::new(array, device)
}

/// ONNX-style GatherND with `batch_dims = 0`, shared by `int_gather_nd` and
/// `float_gather_nd`.
///
/// `indices` is an M-dimensional integer array whose last dimension (size `K`)
/// holds coordinates into the first `K` dims of `data`. The output shape is
/// `indices.shape[..M-1] ++ data.shape[K..]`.
///
/// Implemented in a vectorized (GPU) fashion: flatten `data` to `[P, slice]`
/// where `P = prod(data.shape[..K])`, convert each K-tuple index to a flat row
/// via the first-K-dims strides, then `take` those rows and reshape.
pub(crate) fn gather_nd_array(
    data: &Array,
    data_shape: &[usize],
    indices: &Array,
    idx_shape: &[usize],
) -> Array {
    let m = idx_shape.len();
    let k = idx_shape[m - 1];
    let num_indices: usize = idx_shape[..m - 1].iter().product();
    let slice_size: usize = data_shape[k..].iter().product();
    let p: usize = data_shape[..k].iter().product();

    // Row strides over the first K dims (each "row" is `slice_size` contiguous elems).
    let mut pstrides = vec![0i32; k];
    let mut acc = 1i32;
    for j in (0..k).rev() {
        pstrides[j] = acc;
        acc *= data_shape[j] as i32;
    }

    let data_2d = data
        .reshape(&[p as i32, slice_size as i32])
        .expect("gather_nd: reshape data");
    let idx_2d = indices
        .reshape(&[num_indices as i32, k as i32])
        .expect("gather_nd: reshape indices");
    // flat_row[n] = sum_j idx_2d[n, j] * pstrides[j]
    let pstride_arr = Array::from_slice(&pstrides, &[1, k as i32]);
    let prod = mlx_rs::ops::multiply(&idx_2d, &pstride_arr).expect("gather_nd: multiply");
    let flat = prod.sum_axes(&[1], false).expect("gather_nd: sum");
    let gathered = take_axis(&data_2d, &flat, 0).expect("gather_nd: take");

    let mut out_shape: Vec<i32> = idx_shape[..m - 1].iter().map(|&x| x as i32).collect();
    out_shape.extend(data_shape[k..].iter().map(|&x| x as i32));
    gathered
        .reshape(&out_shape)
        .expect("gather_nd: reshape output")
}
