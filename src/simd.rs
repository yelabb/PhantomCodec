//! SIMD-accelerated operations for neural data processing
//!
//! Provides portable SIMD implementations with fallback to scalar code.
//!
//! **Requires nightly Rust** for SIMD features (unstable `core::simd`).
//!
//! Current support:
//! - ✅ Portable SIMD (core::simd) for host targets (nightly Rust)
//! - ✅ Scalar fallback (works on stable Rust and all targets)
//! - ❌ ARM DSP intrinsics (SADD16, SSUB16) - NOT YET IMPLEMENTED
//! - ❌ Helium (MVE) for Cortex-M55/M85 - planned for future

/// Calculate deltas between consecutive samples: delta[i] = input[i] - input[i-1]
///
/// First element is preserved as-is (no previous value to delta against).
/// Uses SIMD acceleration when available.
///
/// # Example
/// ```
/// # use phantomcodec::simd::compute_deltas;
/// let input = [10, 13, 11, 14, 12];
/// let mut output = [0; 5];
/// compute_deltas(&input, &mut output);
/// assert_eq!(output, [10, 3, -2, 3, -2]);
/// ```
pub fn compute_deltas(input: &[i32], output: &mut [i32]) {
    assert_eq!(input.len(), output.len(), "Input and output must be same length");

    if input.is_empty() {
        return;
    }

    // First element has no previous value
    output[0] = input[0];

    // Dispatch to SIMD or scalar implementation
    #[cfg(all(feature = "simd", target_feature = "simd128"))]
    {
        compute_deltas_simd(&input[1..], &mut output[1..], input[0]);
    }

    #[cfg(not(all(feature = "simd", target_feature = "simd128")))]
    {
        compute_deltas_scalar(&input[1..], &mut output[1..], input[0]);
    }
}

/// Scalar implementation of delta computation
#[inline]
fn compute_deltas_scalar(input: &[i32], output: &mut [i32], mut prev: i32) {
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = *inp - prev;
        prev = *inp;
    }
}

/// SIMD implementation using portable SIMD (8-wide i32 lanes)
#[cfg(all(feature = "simd", target_feature = "simd128"))]
fn compute_deltas_simd(input: &[i32], output: &mut [i32], mut prev: i32) {
    use core::simd::{i32x8, Simd, simd_swizzle};

    const LANES: usize = 8;
    let len = input.len();
    let simd_end = len - (len % LANES);

    let mut i = 0;

    // Process 8 elements at a time
    while i < simd_end {
        let chunk = Simd::<i32, 8>::from_slice(&input[i..i + LANES]);
        
        // Create previous vector: [prev, chunk[0], chunk[1], ..., chunk[6]]
        // Use SIMD shuffle (rotate_elements_right) instead of memory copy
        let prev_vec = simd_swizzle!(Simd::from([prev, 0, 0, 0, 0, 0, 0, 0]), chunk, [
            First(0),  // prev
            Second(0), // chunk[0]
            Second(1), // chunk[1]
            Second(2), // chunk[2]
            Second(3), // chunk[3]
            Second(4), // chunk[4]
            Second(5), // chunk[5]
            Second(6), // chunk[6]
        ]);
        
        // Compute deltas: current - previous
        let deltas = chunk - prev_vec;
        deltas.copy_to_slice(&mut output[i..i + LANES]);
        
        // Update prev for next iteration
        prev = input[i + LANES - 1];
        i += LANES;
    }

    // Handle remaining elements (< 8)
    compute_deltas_scalar(&input[i..], &mut output[i..], prev);
}

/// Reconstruct original values from deltas: output[i] = output[i-1] + delta[i]
///
/// First element is preserved as-is (it's the initial value, not a delta).
///
/// # Example
/// ```
/// # use phantomcodec::simd::reconstruct_from_deltas;
/// let deltas = [10, 3, -2, 3, -2];
/// let mut output = [0; 5];
/// reconstruct_from_deltas(&deltas, &mut output);
/// assert_eq!(output, [10, 13, 11, 14, 12]);
/// ```
pub fn reconstruct_from_deltas(deltas: &[i32], output: &mut [i32]) {
    assert_eq!(deltas.len(), output.len(), "Input and output must be same length");

    if deltas.is_empty() {
        return;
    }

    // First element is the base value
    output[0] = deltas[0];
    let prev = output[0];

    // Dispatch to SIMD or scalar implementation
    #[cfg(all(feature = "simd", target_feature = "simd128"))]
    {
        reconstruct_from_deltas_simd(&deltas[1..], &mut output[1..], prev);
    }

    #[cfg(not(all(feature = "simd", target_feature = "simd128")))]
    {
        reconstruct_from_deltas_scalar(&deltas[1..], &mut output[1..], prev);
    }
}

/// Scalar implementation of delta reconstruction
#[inline]
fn reconstruct_from_deltas_scalar(deltas: &[i32], output: &mut [i32], mut prev: i32) {
    for (delta, out) in deltas.iter().zip(output.iter_mut()) {
        prev += *delta;
        *out = prev;
    }
}

/// SIMD implementation of delta reconstruction
#[cfg(all(feature = "simd", target_feature = "simd128"))]
fn reconstruct_from_deltas_simd(deltas: &[i32], output: &mut [i32], mut prev: i32) {
    use core::simd::{i32x8, Simd};

    const LANES: usize = 8;
    let len = deltas.len();
    let simd_end = len - (len % LANES);

    let mut i = 0;

    // Process 8 elements at a time
    while i < simd_end {
        let delta_chunk = Simd::<i32, 8>::from_slice(&deltas[i..i + LANES]);
        
        // Prefix sum within the SIMD vector
        let mut result = [0i32; LANES];
        result[0] = prev + delta_chunk[0];
        for j in 1..LANES {
            result[j] = result[j - 1] + delta_chunk[j];
        }
        
        output[i..i + LANES].copy_from_slice(&result);
        prev = result[LANES - 1];
        i += LANES;
    }

    // Handle remaining elements
    reconstruct_from_deltas_scalar(&deltas[i..], &mut output[i..], prev);
}

/// Calculate sum of absolute deltas for adaptive Rice parameter selection
///
/// Returns the sum of |delta[i]| for the first `n` samples (or all if n > len).
/// This heuristic is used to choose Rice parameter k:
/// - High sum → high activity → use k=3
/// - Low sum → low activity → use k=1
///
/// # Example
/// ```
/// # use phantomcodec::simd::sum_abs_deltas;
/// let deltas = [10, 3, -2, 3, -2, 1, -1, 0];
/// let sum = sum_abs_deltas(&deltas, 16);
/// assert_eq!(sum, 10 + 3 + 2 + 3 + 2 + 1 + 1 + 0);
/// ```
pub fn sum_abs_deltas(deltas: &[i32], n: usize) -> u32 {
    let n = n.min(deltas.len());
    
    #[cfg(all(feature = "simd", target_feature = "simd128"))]
    {
        sum_abs_deltas_simd(&deltas[..n])
    }

    #[cfg(not(all(feature = "simd", target_feature = "simd128")))]
    {
        sum_abs_deltas_scalar(&deltas[..n])
    }
}

/// Scalar implementation of sum of absolute deltas
#[inline]
fn sum_abs_deltas_scalar(deltas: &[i32]) -> u32 {
    deltas.iter().map(|&x| x.abs() as u32).sum()
}

/// SIMD implementation of sum of absolute deltas
#[cfg(all(feature = "simd", target_feature = "simd128"))]
fn sum_abs_deltas_simd(deltas: &[i32]) -> u32 {
    use core::simd::{i32x8, Simd, SimdInt};

    const LANES: usize = 8;
    let len = deltas.len();
    let simd_end = len - (len % LANES);

    let mut sum_vec = Simd::<i32, 8>::splat(0);
    let mut i = 0;

    // Process 8 elements at a time
    while i < simd_end {
        let chunk = Simd::<i32, 8>::from_slice(&deltas[i..i + LANES]);
        let abs_chunk = chunk.abs();
        sum_vec += abs_chunk;
        i += LANES;
    }

    // Horizontal sum of SIMD vector
    let mut total = sum_vec.to_array().iter().sum::<i32>() as u32;

    // Add remaining elements
    total += sum_abs_deltas_scalar(&deltas[i..]);

    total
}

// ARM Cortex-M DSP intrinsics (NOT YET IMPLEMENTED)
#[cfg(feature = "cortex-m-dsp")]
mod cortex_m_dsp {
    // ⚠️ WARNING: This module is currently EMPTY
    //
    // Despite earlier claims in documentation, ARM DSP intrinsics (SADD16, SSUB16)
    // for Cortex-M4F are not yet implemented. Code currently falls back to scalar
    // operations on Cortex-M targets.
    //
    // To implement (future work):
    // 1. Use cortex-m crate's DSP intrinsics (__SSUB16 for dual 16-bit subtraction)
    // 2. Convert i32 to Q15 format (pack two samples into one u32 register)
    // 3. Perform parallel operations using SIMD instructions
    // 4. Unpack results back to i32
    //
    // Expected speedup: 1.5-2x for Cortex-M4F over scalar code
    // Until then: Use portable SIMD on host targets, or scalar fallback on embedded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_deltas() {
        let input = [10, 13, 11, 14, 12];
        let mut output = [0; 5];
        compute_deltas(&input, &mut output);
        assert_eq!(output, [10, 3, -2, 3, -2]);
    }

    #[test]
    fn test_compute_deltas_empty() {
        let input: [i32; 0] = [];
        let mut output: [i32; 0] = [];
        compute_deltas(&input, &mut output);
        assert_eq!(output, []);
    }

    #[test]
    fn test_compute_deltas_single() {
        let input = [42];
        let mut output = [0];
        compute_deltas(&input, &mut output);
        assert_eq!(output, [42]);
    }

    #[test]
    fn test_reconstruct_from_deltas() {
        let deltas = [10, 3, -2, 3, -2];
        let mut output = [0; 5];
        reconstruct_from_deltas(&deltas, &mut output);
        assert_eq!(output, [10, 13, 11, 14, 12]);
    }

    #[test]
    fn test_roundtrip() {
        let original = [5, 8, 6, 9, 7, 10, 8];
        let mut deltas = [0; 7];
        let mut reconstructed = [0; 7];

        compute_deltas(&original, &mut deltas);
        reconstruct_from_deltas(&deltas, &mut reconstructed);

        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_sum_abs_deltas() {
        let deltas = [10, 3, -2, 3, -2, 1, -1, 0];
        let sum = sum_abs_deltas(&deltas, 16);
        assert_eq!(sum, 10 + 3 + 2 + 3 + 2 + 1 + 1 + 0);
    }

    #[test]
    fn test_sum_abs_deltas_partial() {
        let deltas = [10, 3, -2, 3, -2, 1, -1, 0];
        let sum = sum_abs_deltas(&deltas, 4);
        assert_eq!(sum, 10 + 3 + 2 + 3);
    }

    #[test]
    fn test_large_dataset() {
        // Test with realistic neural data size
        let input: Vec<i32> = (0..1024).map(|i| (i % 10) as i32).collect();
        let mut deltas = vec![0; 1024];
        let mut reconstructed = vec![0; 1024];

        compute_deltas(&input, &mut deltas);
        reconstruct_from_deltas(&deltas, &mut reconstructed);

        assert_eq!(input, reconstructed);
    }
}
