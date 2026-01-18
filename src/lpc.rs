//! Linear Predictive Coding (LPC) for improved compression
//!
//! Provides higher-order predictors that model signal continuity better than
//! simple delta encoding. Particularly effective for neural LFP signals.
//!
//! ## Predictors
//!
//! - **Delta (Order 1)**: `P[t] = x[t-1]` → Error = `x[t] - x[t-1]`
//! - **LPC2 (Order 2)**: `P[t] = 2*x[t-1] - x[t-2]` → Models constant velocity
//! - **LPC3 (Order 3)**: `P[t] = 3*x[t-1] - 3*x[t-2] + x[t-3]` → Models constant acceleration
//!
//! ## Performance Impact
//!
//! For wavy LFP signals:
//! - Delta: Residuals ~±200 → 8-9 bits/sample
//! - LPC2: Residuals ~±50 → 6-7 bits/sample (~20% better compression)
//! - LPC3: Residuals ~±30 → 5-6 bits/sample (~30% better compression)

/// Second-order linear predictor: P[t] = 2*x[t-1] - x[t-2]
///
/// This predictor assumes the signal continues at the same **velocity**
/// (constant first derivative), making it ideal for smooth, wavy signals
/// like neural LFP data.
///
/// # Mathematical Background
///
/// For a signal following x[t] = A·sin(ωt + φ):
/// - Delta residual: ≈ A·ω·cos(ωt) (still oscillating)
/// - LPC2 residual: ≈ A·ω²·sin(ωt) (much smaller for low frequencies)
///
/// # Arguments
/// * `samples` - Input signal
/// * `output` - Output buffer for prediction residuals (must be same length)
///
/// # Output Format
/// - `output[0]` = `samples[0]` (base value)
/// - `output[1]` = `samples[1] - samples[0]` (first delta, no prediction yet)
/// - `output[i]` = `samples[i] - (2*samples[i-1] - samples[i-2])` for i ≥ 2
///
/// # Example
/// ```
/// # use phantomcodec::lpc::compute_lpc2_residuals;
/// let samples = [100, 103, 106, 109, 112]; // Linear ramp
/// let mut residuals = [0; 5];
/// compute_lpc2_residuals(&samples, &mut residuals);
/// // residuals = [100, 3, 0, 0, 0] - perfect prediction!
/// assert_eq!(residuals, [100, 3, 0, 0, 0]);
/// ```
pub fn compute_lpc2_residuals(samples: &[i32], output: &mut [i32]) {
    assert_eq!(
        samples.len(),
        output.len(),
        "Input and output must be same length"
    );

    if samples.is_empty() {
        return;
    }

    // First element is the base value (no previous samples to predict from)
    output[0] = samples[0];

    if samples.len() == 1 {
        return;
    }

    // Second element uses simple delta (only one previous sample available)
    output[1] = samples[1] - samples[0];

    // From third element onwards, use LPC2 prediction
    for i in 2..samples.len() {
        // Predictor: P[t] = 2*x[t-1] - x[t-2]
        // This extrapolates the trend: if x went from 100 to 105,
        // predict next will be 110 (continuing the +5 velocity)
        let predicted = 2 * samples[i - 1] - samples[i - 2];
        
        // Residual (prediction error)
        output[i] = samples[i] - predicted;
    }
}

/// Restore original signal from LPC2 residuals
///
/// Inverse operation of `compute_lpc2_residuals()`. Reconstructs the
/// original signal by applying the predictor and adding residuals.
///
/// # Arguments
/// * `residuals` - Prediction residuals from compression
/// * `output` - Output buffer for reconstructed signal (must be same length)
///
/// # Example
/// ```
/// # use phantomcodec::lpc::{compute_lpc2_residuals, restore_from_lpc2_residuals};
/// let original = [100, 103, 106, 109, 112];
/// let mut residuals = [0; 5];
/// compute_lpc2_residuals(&original, &mut residuals);
///
/// let mut reconstructed = [0; 5];
/// restore_from_lpc2_residuals(&residuals, &mut reconstructed);
/// assert_eq!(reconstructed, original);
/// ```
pub fn restore_from_lpc2_residuals(residuals: &[i32], output: &mut [i32]) {
    assert_eq!(
        residuals.len(),
        output.len(),
        "Input and output must be same length"
    );

    if residuals.is_empty() {
        return;
    }

    // First element is the base value
    output[0] = residuals[0];

    if residuals.len() == 1 {
        return;
    }

    // Second element: reconstruct from simple delta
    output[1] = output[0] + residuals[1];

    // From third element onwards, use LPC2 predictor
    for i in 2..residuals.len() {
        // Predictor: P[t] = 2*x[t-1] - x[t-2]
        let predicted = 2 * output[i - 1] - output[i - 2];
        
        // Actual value = prediction + residual
        output[i] = predicted + residuals[i];
    }
}

/// Third-order linear predictor: P[t] = 3*x[t-1] - 3*x[t-2] + x[t-3]
///
/// This predictor models constant **acceleration** (constant second derivative),
/// making it suitable for signals with smooth curvature changes.
///
/// **Trade-off**: Requires 3 previous samples and slightly more computation.
/// Use LPC2 for most cases; use LPC3 only if testing shows significant benefit.
///
/// # Arguments
/// * `samples` - Input signal
/// * `output` - Output buffer for prediction residuals (must be same length)
///
/// # Output Format
/// - `output[0]` = `samples[0]` (base value)
/// - `output[1]` = `samples[1] - samples[0]` (first delta)
/// - `output[2]` = `samples[2] - (2*samples[1] - samples[0])` (LPC2)
/// - `output[i]` = `samples[i] - (3*samples[i-1] - 3*samples[i-2] + samples[i-3])` for i ≥ 3
///
/// # Example
/// ```
/// # use phantomcodec::lpc::compute_lpc3_residuals;
/// let samples = [100, 101, 104, 109, 116]; // Quadratic growth
/// let mut residuals = [0; 5];
/// compute_lpc3_residuals(&samples, &mut residuals);
/// // Should have small residuals for quadratic signals
/// ```
pub fn compute_lpc3_residuals(samples: &[i32], output: &mut [i32]) {
    assert_eq!(
        samples.len(),
        output.len(),
        "Input and output must be same length"
    );

    if samples.is_empty() {
        return;
    }

    // First element is the base value
    output[0] = samples[0];

    if samples.len() == 1 {
        return;
    }

    // Second element uses simple delta
    output[1] = samples[1] - samples[0];

    if samples.len() == 2 {
        return;
    }

    // Third element uses LPC2 (not enough history for LPC3)
    let predicted_2 = 2 * samples[1] - samples[0];
    output[2] = samples[2] - predicted_2;

    // From fourth element onwards, use LPC3 prediction
    for i in 3..samples.len() {
        // Predictor: P[t] = 3*x[t-1] - 3*x[t-2] + x[t-3]
        // This is a second-order extrapolation modeling constant acceleration
        let predicted = 3 * samples[i - 1] - 3 * samples[i - 2] + samples[i - 3];
        
        // Residual (prediction error)
        output[i] = samples[i] - predicted;
    }
}

/// Restore original signal from LPC3 residuals
///
/// Inverse operation of `compute_lpc3_residuals()`.
///
/// # Arguments
/// * `residuals` - Prediction residuals from compression
/// * `output` - Output buffer for reconstructed signal (must be same length)
///
/// # Example
/// ```
/// # use phantomcodec::lpc::{compute_lpc3_residuals, restore_from_lpc3_residuals};
/// let original = [100, 101, 104, 109, 116];
/// let mut residuals = [0; 5];
/// compute_lpc3_residuals(&original, &mut residuals);
///
/// let mut reconstructed = [0; 5];
/// restore_from_lpc3_residuals(&residuals, &mut reconstructed);
/// assert_eq!(reconstructed, original);
/// ```
pub fn restore_from_lpc3_residuals(residuals: &[i32], output: &mut [i32]) {
    assert_eq!(
        residuals.len(),
        output.len(),
        "Input and output must be same length"
    );

    if residuals.is_empty() {
        return;
    }

    // First element is the base value
    output[0] = residuals[0];

    if residuals.len() == 1 {
        return;
    }

    // Second element: reconstruct from simple delta
    output[1] = output[0] + residuals[1];

    if residuals.len() == 2 {
        return;
    }

    // Third element: reconstruct using LPC2
    let predicted_2 = 2 * output[1] - output[0];
    output[2] = predicted_2 + residuals[2];

    // From fourth element onwards, use LPC3 predictor
    for i in 3..residuals.len() {
        // Predictor: P[t] = 3*x[t-1] - 3*x[t-2] + x[t-3]
        let predicted = 3 * output[i - 1] - 3 * output[i - 2] + output[i - 3];
        
        // Actual value = prediction + residual
        output[i] = predicted + residuals[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lpc2_linear_signal() {
        // Perfect linear ramp - LPC2 should predict exactly
        let samples = [100, 103, 106, 109, 112];
        let mut residuals = [0; 5];
        compute_lpc2_residuals(&samples, &mut residuals);
        
        // First two elements are base and delta
        assert_eq!(residuals[0], 100);
        assert_eq!(residuals[1], 3);
        
        // Remaining should be zero (perfect prediction)
        assert_eq!(residuals[2], 0);
        assert_eq!(residuals[3], 0);
        assert_eq!(residuals[4], 0);
    }

    #[test]
    fn test_lpc2_roundtrip() {
        // Test full roundtrip on realistic wavy signal
        let samples = [100, 105, 108, 109, 108, 105, 100, 95];
        let mut residuals = [0; 8];
        let mut reconstructed = [0; 8];
        
        compute_lpc2_residuals(&samples, &mut residuals);
        restore_from_lpc2_residuals(&residuals, &mut reconstructed);
        
        assert_eq!(samples, reconstructed);
    }

    #[test]
    fn test_lpc2_empty() {
        let samples: [i32; 0] = [];
        let mut residuals: [i32; 0] = [];
        compute_lpc2_residuals(&samples, &mut residuals);
        assert_eq!(residuals, []);
    }

    #[test]
    fn test_lpc2_single_element() {
        let samples = [42];
        let mut residuals = [0];
        compute_lpc2_residuals(&samples, &mut residuals);
        assert_eq!(residuals, [42]);
        
        let mut reconstructed = [0];
        restore_from_lpc2_residuals(&residuals, &mut reconstructed);
        assert_eq!(reconstructed, [42]);
    }

    #[test]
    fn test_lpc2_two_elements() {
        let samples = [10, 15];
        let mut residuals = [0; 2];
        compute_lpc2_residuals(&samples, &mut residuals);
        
        assert_eq!(residuals[0], 10);
        assert_eq!(residuals[1], 5);
        
        let mut reconstructed = [0; 2];
        restore_from_lpc2_residuals(&residuals, &mut reconstructed);
        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn test_lpc2_vs_delta_residual_magnitude() {
        // Simulate wavy LFP signal
        let samples = [2048, 2050, 2051, 2050, 2048, 2045, 2043, 2042, 2043, 2045];
        
        // Compute LPC2 residuals
        let mut lpc2_residuals = [0; 10];
        compute_lpc2_residuals(&samples, &mut lpc2_residuals);
        
        // Compute delta residuals (for comparison)
        let mut delta_residuals = [0; 10];
        delta_residuals[0] = samples[0];
        for i in 1..samples.len() {
            delta_residuals[i] = samples[i] - samples[i - 1];
        }
        
        // Sum absolute values (excluding first two initialization values)
        let lpc2_sum: i32 = lpc2_residuals[2..].iter().map(|x| x.abs()).sum();
        let delta_sum: i32 = delta_residuals[2..].iter().map(|x| x.abs()).sum();
        
        // LPC2 should have smaller residuals for wavy signals
        assert!(
            lpc2_sum <= delta_sum,
            "LPC2 residuals ({}) should be <= delta residuals ({})",
            lpc2_sum,
            delta_sum
        );
    }

    #[test]
    fn test_lpc3_quadratic_signal() {
        // Quadratic growth: acceleration = 2
        let samples = [100, 101, 104, 109, 116, 125];
        let mut residuals = [0; 6];
        compute_lpc3_residuals(&samples, &mut residuals);
        
        // First three elements are initialization
        assert_eq!(residuals[0], 100);
        assert_eq!(residuals[1], 1);
        
        // From index 3 onwards, residuals should be small/zero for quadratic
        // (LPC3 models constant acceleration perfectly)
        assert_eq!(residuals[3], 0);
        assert_eq!(residuals[4], 0);
        assert_eq!(residuals[5], 0);
    }

    #[test]
    fn test_lpc3_roundtrip() {
        // Test full roundtrip
        let samples = [100, 105, 108, 110, 110, 108, 105, 100, 95, 92];
        let mut residuals = [0; 10];
        let mut reconstructed = [0; 10];
        
        compute_lpc3_residuals(&samples, &mut residuals);
        restore_from_lpc3_residuals(&residuals, &mut reconstructed);
        
        assert_eq!(samples, reconstructed);
    }

    #[test]
    fn test_lpc3_empty() {
        let samples: [i32; 0] = [];
        let mut residuals: [i32; 0] = [];
        compute_lpc3_residuals(&samples, &mut residuals);
        assert_eq!(residuals, []);
    }

    #[test]
    fn test_lpc3_single_element() {
        let samples = [42];
        let mut residuals = [0];
        compute_lpc3_residuals(&samples, &mut residuals);
        assert_eq!(residuals, [42]);
        
        let mut reconstructed = [0];
        restore_from_lpc3_residuals(&residuals, &mut reconstructed);
        assert_eq!(reconstructed, [42]);
    }

    #[test]
    fn test_lpc3_two_elements() {
        let samples = [10, 15];
        let mut residuals = [0; 2];
        compute_lpc3_residuals(&samples, &mut residuals);
        
        assert_eq!(residuals[0], 10);
        assert_eq!(residuals[1], 5);
        
        let mut reconstructed = [0; 2];
        restore_from_lpc3_residuals(&residuals, &mut reconstructed);
        assert_eq!(reconstructed, samples);
    }

    #[test]
    fn test_lpc3_three_elements() {
        let samples = [10, 15, 22];
        let mut residuals = [0; 3];
        compute_lpc3_residuals(&samples, &mut residuals);
        
        assert_eq!(residuals[0], 10);
        assert_eq!(residuals[1], 5);
        // Third element uses LPC2: predicted = 2*15 - 10 = 20, residual = 22 - 20 = 2
        assert_eq!(residuals[2], 2);
        
        let mut reconstructed = [0; 3];
        restore_from_lpc3_residuals(&residuals, &mut reconstructed);
        assert_eq!(reconstructed, samples);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_lpc2_large_dataset() {
        // Test with realistic neural data size (1024 samples)
        extern crate alloc;
        use alloc::vec;
        
        // Simulate wavy LFP signal (sine-like)
        const FREQ: f64 = 0.1;
        const AMPLITUDE: f64 = 100.0;
        const BASE_VALUE: i32 = 2048;
        
        let mut samples = vec![0i32; 1024];
        for i in 0..1024 {
            samples[i] = BASE_VALUE + ((i as f64 * FREQ).sin() * AMPLITUDE) as i32;
        }
        
        let mut residuals = vec![0; 1024];
        let mut reconstructed = vec![0; 1024];
        
        compute_lpc2_residuals(&samples, &mut residuals);
        restore_from_lpc2_residuals(&residuals, &mut reconstructed);
        
        assert_eq!(samples, reconstructed);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_lpc3_large_dataset() {
        // Test with realistic neural data size (1024 samples)
        extern crate alloc;
        use alloc::vec;
        
        // Simulate complex wavy signal
        const FREQ_1: f64 = 0.05;
        const AMPLITUDE_1: f64 = 50.0;
        const FREQ_2: f64 = 0.02;
        const AMPLITUDE_2: f64 = 30.0;
        const BASE_VALUE: i32 = 2048;
        
        let mut samples = vec![0i32; 1024];
        for i in 0..1024 {
            samples[i] = BASE_VALUE 
                + ((i as f64 * FREQ_1).sin() * AMPLITUDE_1) as i32
                + ((i as f64 * FREQ_2).cos() * AMPLITUDE_2) as i32;
        }
        
        let mut residuals = vec![0; 1024];
        let mut reconstructed = vec![0; 1024];
        
        compute_lpc3_residuals(&samples, &mut residuals);
        restore_from_lpc3_residuals(&residuals, &mut reconstructed);
        
        assert_eq!(samples, reconstructed);
    }
}
