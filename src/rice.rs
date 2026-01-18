//! Adaptive Rice coding implementation
//!
//! Golomb-Rice coding is a family of entropy coding schemes optimal for
//! geometric distributions (common in delta-encoded neural data).
//! 
//! The Rice parameter `k` determines the split between unary and binary parts:
//! - Small k (0-1): Optimized for values near 0 (sparse activity)
//! - Large k (2-3): Optimized for larger values (burst activity)
//!
//! We adaptively select k per frame using a Mean Absolute Deviation (MAD)
//! heuristic on the first 16 samples.

use crate::bitwriter::{BitReader, BitWriter};
use crate::error::{CodecError, CodecResult};
use crate::simd::sum_abs_deltas;

/// Maximum Rice parameter value (2 bits in header)
pub const MAX_RICE_K: u8 = 3;

/// Number of samples to analyze for adaptive k selection
const ADAPTIVE_SAMPLE_SIZE: usize = 16;

/// Threshold for switching from k=1 to k=3
/// If sum of absolute deltas > threshold, use k=3 (high activity)
const HIGH_ACTIVITY_THRESHOLD: u32 = 48;

/// Select optimal Rice parameter k based on first 16 deltas
///
/// Uses a simplified MAD heuristic:
/// - Calculate sum of absolute deltas for first 16 samples
/// - If sum > 48 (mean of ~3 per sample), use k=3 (burst mode)
/// - Otherwise use k=1 (sparse mode)
///
/// # Example
/// ```
/// # use phantomcodec::rice::select_rice_parameter;
/// let sparse_deltas = [0, 1, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0];
/// let k = select_rice_parameter(&sparse_deltas);
/// assert_eq!(k, 1); // Low activity
///
/// let burst_deltas = [5, 7, -4, 8, 6, -5, 9, 7, -6, 8, 5, -7, 6, 8, -5, 7];
/// let k = select_rice_parameter(&burst_deltas);
/// assert_eq!(k, 3); // High activity
/// ```
pub fn select_rice_parameter(deltas: &[i32]) -> u8 {
    let n = deltas.len().min(ADAPTIVE_SAMPLE_SIZE);
    if n == 0 {
        return 1; // Default for empty input
    }

    let sum = sum_abs_deltas(deltas, n);
    
    if sum > HIGH_ACTIVITY_THRESHOLD {
        3 // High activity: use larger k
    } else {
        1 // Low activity: use smaller k
    }
}

/// Encode a single value using Rice coding
///
/// Rice coding splits value into:
/// - Quotient q = value >> k (encoded in unary: q zeros + 1)
/// - Remainder r = value & ((1 << k) - 1) (encoded in binary using k bits)
///
/// # Arguments
/// * `writer` - BitWriter to write encoded bits
/// * `value` - Unsigned value to encode (must be non-negative)
/// * `k` - Rice parameter (0-3)
///
/// # Example
/// ```
/// # use phantomcodec::rice::rice_encode;
/// # use phantomcodec::bitwriter::BitWriter;
/// let mut buffer = [0u8; 10];
/// let mut writer = BitWriter::new(&mut buffer);
/// 
/// rice_encode(&mut writer, 7, 2).unwrap();
/// // 7 >> 2 = 1 (quotient) → unary: 01
/// // 7 & 3 = 3 (remainder) → binary: 11
/// // Result: 0111
/// ```
pub fn rice_encode(writer: &mut BitWriter, value: u32, k: u8) -> CodecResult<()> {
    if k > MAX_RICE_K {
        return Err(CodecError::InvalidRiceParameter { k });
    }

    // Split into quotient and remainder
    let quotient = value >> k;
    let remainder = value & ((1u32 << k) - 1);

    // Encode quotient in unary (q zeros followed by a 1)
    // Clamp quotient to prevent excessive unary length
    let quotient_clamped = quotient.min(255); // Max 255 zeros
    for _ in 0..quotient_clamped {
        writer.write_bits(0, 1)?;
    }
    writer.write_bits(1, 1)?;

    // Encode remainder in binary (k bits)
    if k > 0 {
        writer.write_bits(remainder, k)?;
    }

    Ok(())
}

/// Decode a single value using Rice coding
///
/// # Arguments
/// * `reader` - BitReader to read encoded bits
/// * `k` - Rice parameter (must match encoding)
///
/// # Returns
/// Decoded unsigned value
pub fn rice_decode(reader: &mut BitReader, k: u8) -> CodecResult<u32> {
    if k > MAX_RICE_K {
        return Err(CodecError::InvalidRiceParameter { k });
    }

    // Decode unary quotient (count zeros until we hit a 1)
    let mut quotient = 0u32;
    loop {
        let bit = reader.read_bits(1)?;
        if bit == 1 {
            break;
        }
        quotient += 1;

        // Prevent infinite loop on corrupted data
        if quotient > 1024 {
            return Err(CodecError::UnexpectedEndOfInput);
        }
    }

    // Decode binary remainder (k bits)
    let remainder = if k > 0 {
        reader.read_bits(k)?
    } else {
        0
    };

    // Reconstruct value
    let value = (quotient << k) | remainder;
    Ok(value)
}

/// Encode array of deltas using adaptive Rice coding
///
/// # Arguments
/// * `deltas` - Delta-encoded values (first element is base value)
/// * `output` - Output buffer for compressed bits
/// * `use_zigzag` - Whether to apply ZigZag encoding for signed values
///
/// # Returns
/// Tuple of (bytes_written, selected_k)
pub fn rice_encode_array(
    deltas: &[i32],
    output: &mut [u8],
    use_zigzag: bool,
) -> CodecResult<(usize, u8)> {
    if deltas.is_empty() {
        return Ok((0, 1));
    }

    // Select Rice parameter adaptively
    let k = select_rice_parameter(deltas);

    let mut writer = BitWriter::new(output);

    // Encode each delta
    for &delta in deltas {
        let value = if use_zigzag {
            // ZigZag encode signed value
            ((delta << 1) ^ (delta >> 31)) as u32
        } else {
            // Treat as unsigned (spike counts are always positive)
            delta as u32
        };

        rice_encode(&mut writer, value, k)?;
    }

    writer.flush()?;
    Ok((writer.bytes_written(), k))
}

/// Decode array of deltas using Rice coding
///
/// # Arguments
/// * `input` - Compressed bit stream
/// * `output` - Output buffer for decoded deltas
/// * `k` - Rice parameter (from packet header)
/// * `use_zigzag` - Whether to apply ZigZag decoding
///
/// # Returns
/// Number of values decoded
pub fn rice_decode_array(
    input: &[u8],
    output: &mut [i32],
    k: u8,
    use_zigzag: bool,
) -> CodecResult<usize> {
    let mut reader = BitReader::new(input);
    let mut count = 0;

    while count < output.len() && reader.remaining_bits() > 0 {
        let value = rice_decode(&mut reader, k)?;

        let decoded = if use_zigzag {
            // ZigZag decode to signed
            ((value >> 1) as i32) ^ (-((value & 1) as i32))
        } else {
            value as i32
        };

        output[count] = decoded;
        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rice_encode_decode() {
        let mut buffer = [0u8; 100];
        let mut writer = BitWriter::new(&mut buffer);

        let values = [0, 1, 2, 3, 7, 15, 31];
        let k = 2;

        for &val in &values {
            rice_encode(&mut writer, val, k).unwrap();
        }
        writer.flush().unwrap();

        let mut reader = BitReader::new(&buffer);
        for &expected in &values {
            let decoded = rice_decode(&mut reader, k).unwrap();
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn test_select_rice_parameter_sparse() {
        let sparse = [0, 1, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0];
        let k = select_rice_parameter(&sparse);
        assert_eq!(k, 1);
    }

    #[test]
    fn test_select_rice_parameter_burst() {
        let burst = [5, 7, -4, 8, 6, -5, 9, 7, -6, 8, 5, -7, 6, 8, -5, 7];
        let k = select_rice_parameter(&burst);
        assert_eq!(k, 3);
    }

    #[test]
    fn test_rice_encode_array_no_zigzag() {
        let deltas = [10, 3, 2, 3, 2, 1, 1, 0];
        let mut buffer = [0u8; 100];

        let (bytes_written, k) = rice_encode_array(&deltas, &mut buffer, false).unwrap();
        assert!(bytes_written > 0);
        assert!(k >= 1 && k <= 3);

        let mut decoded = [0i32; 8];
        let count = rice_decode_array(&buffer[..bytes_written], &mut decoded, k, false).unwrap();
        assert_eq!(count, 8);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_rice_encode_array_with_zigzag() {
        let deltas = [10, 3, -2, 3, -2, 1, -1, 0];
        let mut buffer = [0u8; 100];

        let (bytes_written, k) = rice_encode_array(&deltas, &mut buffer, true).unwrap();
        assert!(bytes_written > 0);

        let mut decoded = [0i32; 8];
        let count = rice_decode_array(&buffer[..bytes_written], &mut decoded, k, true).unwrap();
        assert_eq!(count, 8);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_rice_invalid_k() {
        let mut buffer = [0u8; 10];
        let mut writer = BitWriter::new(&mut buffer);

        let result = rice_encode(&mut writer, 7, 5); // k=5 is invalid
        assert!(matches!(result, Err(CodecError::InvalidRiceParameter { k: 5 })));
    }

    #[test]
    fn test_rice_empty_array() {
        let deltas: [i32; 0] = [];
        let mut buffer = [0u8; 10];

        let result = rice_encode_array(&deltas, &mut buffer, false).unwrap();
        assert_eq!(result, (0, 1));
    }
}
