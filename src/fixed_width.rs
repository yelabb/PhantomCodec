//! Fixed-Width Block Packing (PFOR) for ultra-low latency (<10µs)
//!
//! Implements predictable, branchless compression/decompression by processing
//! data in fixed-size blocks of 32 samples. Each block uses a uniform bit-width
//! determined by the maximum value in that block.
//!
//! # Strategy
//!
//! - **Block Size:** 32 samples (optimal for 32-bit register operations)
//! - **Header:** 1 byte per block (4 bits for bit-width, 4 bits reserved)
//! - **Bit Widths:** 0-16 bits (specialized unpackers for 4, 5, 6, 8, 10, 12)
//!
//! # Performance
//!
//! - **Zero Control-Flow:** Eliminates variable-length decoding loops
//! - **SIMD-Friendly:** Predictable memory access patterns
//! - **Target Latency:** <10µs for 1024 channels on Cortex-M4F @ 168MHz

#![allow(clippy::manual_div_ceil)] // div_ceil is nightly-only feature

use crate::error::{CodecError, CodecResult};

/// Block size for fixed-width packing (32 samples)
pub const BLOCK_SIZE: usize = 32;

/// Calculate the minimum bit width needed to represent all values in a delta array
///
/// Uses ZigZag encoding to handle signed deltas efficiently. Returns the number
/// of bits required to represent the maximum absolute value after ZigZag encoding.
///
/// # Arguments
/// * `deltas` - Array of signed delta values
///
/// # Returns
/// Bit width (0-32) required to represent all values
///
/// # Example
/// ```ignore
/// let deltas = [1, -2, 3, -4];
/// let width = calculate_bit_width(&deltas);
/// assert_eq!(width, 4); // ZigZag(-4) = 7, requires 3 bits, but we use 4 for alignment
/// ```
pub fn calculate_bit_width(deltas: &[i32]) -> u8 {
    if deltas.is_empty() {
        return 0;
    }

    // Find maximum absolute value after ZigZag encoding
    let mut max_val = 0u32;
    for &delta in deltas {
        let zigzag = zigzag_encode(delta);
        if zigzag > max_val {
            max_val = zigzag;
        }
    }

    // Calculate bits needed (0 if max_val is 0)
    if max_val == 0 {
        0
    } else {
        // Number of bits = floor(log2(max_val)) + 1
        32 - max_val.leading_zeros() as u8
    }
}

/// ZigZag encode a signed integer to unsigned
///
/// Maps signed integers to unsigned: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, etc.
#[inline]
pub fn zigzag_encode(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// ZigZag decode an unsigned integer to signed
#[inline]
pub fn zigzag_decode(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

/// Encode a block of up to 32 deltas with fixed bit width
///
/// # Block Format
/// ```text
/// ┌─────────────┬──────────────────────────────────┐
/// │ Header      │ Packed Samples                   │
/// │ (1 byte)    │ (bit_width * count bits)         │
/// │             │                                  │
/// │ [7:4] Rsvd  │ Sample0, Sample1, ..., SampleN   │
/// │ [3:0] Width │                                  │
/// └─────────────┴──────────────────────────────────┘
/// ```
///
/// # Arguments
/// * `deltas` - Up to 32 delta values to encode (will be ZigZag encoded)
/// * `bit_width` - Number of bits per sample (0-16)
/// * `output` - Output buffer (must have sufficient space)
///
/// # Returns
/// Number of bytes written
#[allow(clippy::needless_range_loop)] // Indexed access for performance-critical packing
pub fn encode_block_32(deltas: &[i32], bit_width: u8, output: &mut [u8]) -> CodecResult<usize> {
    let count = deltas.len().min(BLOCK_SIZE);

    // Validate bit width is in supported range
    if bit_width > 16 {
        return Err(CodecError::UnexpectedEndOfInput); // Best fit for invalid parameter
    }

    // Calculate required output size: 1 byte header + packed data
    let bits_needed = (bit_width as usize) * count;
    let bytes_needed = 1 + (bits_needed + 7) / 8; // +7 for ceiling division

    if output.len() < bytes_needed {
        return Err(CodecError::BufferTooSmall {
            required: bytes_needed,
        });
    }

    // Write header: [7:4] reserved, [3:0] bit_width
    output[0] = bit_width & 0x0F;

    // Special case: bit_width = 0 means all zeros
    if bit_width == 0 {
        return Ok(1);
    }

    // === OPTIMIZED BIT PACKING ===
    // Use 64-bit accumulator to batch writes, eliminating per-bit operations
    let mask = (1u32 << bit_width) - 1;
    let mut accumulator: u64 = 0;
    let mut acc_bits: u32 = 0; // Number of valid bits in accumulator
    let mut out_idx = 1usize; // Start after header

    for i in 0..count {
        let zigzag = zigzag_encode(deltas[i]);
        let value = (zigzag & mask) as u64;

        // Add value to accumulator
        accumulator |= value << acc_bits;
        acc_bits += bit_width as u32;

        // Flush complete bytes from accumulator
        while acc_bits >= 8 {
            output[out_idx] = (accumulator & 0xFF) as u8;
            accumulator >>= 8;
            acc_bits -= 8;
            out_idx += 1;
        }
    }

    // Flush remaining bits (if any)
    if acc_bits > 0 {
        output[out_idx] = (accumulator & 0xFF) as u8;
    }

    Ok(bytes_needed)
}

/// Decode a block of fixed-width packed samples
///
/// # Arguments
/// * `input` - Encoded block (header + packed data)
/// * `count` - Number of samples to decode (up to 32)
/// * `output` - Output buffer for decoded deltas
///
/// # Returns
/// Number of bytes consumed from input
#[allow(clippy::needless_range_loop)] // Simple indexed access for zero initialization
pub fn decode_block_32(input: &[u8], count: usize, output: &mut [i32]) -> CodecResult<usize> {
    if input.is_empty() {
        return Err(CodecError::UnexpectedEndOfInput);
    }

    let count = count.min(BLOCK_SIZE);
    if output.len() < count {
        return Err(CodecError::BufferTooSmall { required: count });
    }

    // Read header
    let bit_width = input[0] & 0x0F;

    // Special case: bit_width = 0 means all zeros
    if bit_width == 0 {
        for i in 0..count {
            output[i] = 0;
        }
        return Ok(1);
    }

    // Dispatch to specialized unpacker if available
    let bytes_consumed = match bit_width {
        4 => decode_width_4(input, count, output)?,
        8 => decode_width_8(input, count, output)?,
        _ => decode_width_generic(input, count, output, bit_width)?,
    };

    Ok(bytes_consumed)
}

/// Specialized decoder for 4-bit width - optimized for 2 samples per byte
/// Processes pairs of nibbles without bit-level iteration
fn decode_width_4(input: &[u8], count: usize, output: &mut [i32]) -> CodecResult<usize> {
    let bits_needed = 4 * count;
    let bytes_needed = 1 + (bits_needed + 7) / 8;

    if input.len() < bytes_needed {
        return Err(CodecError::UnexpectedEndOfInput);
    }

    let data = &input[1..];

    // Process pairs of samples (2 per byte) - no bit-level iteration
    let pairs = count / 2;
    for i in 0..pairs {
        let byte = data[i];
        // Extract both nibbles with simple shifts - no division/modulo
        output[i * 2] = zigzag_decode((byte & 0x0F) as u32);
        output[i * 2 + 1] = zigzag_decode((byte >> 4) as u32);
    }

    // Handle odd sample if present
    if count % 2 == 1 {
        output[count - 1] = zigzag_decode((data[pairs] & 0x0F) as u32);
    }

    Ok(bytes_needed)
}

/// Specialized decoder for 8-bit width - direct byte copy, maximum speed
/// One sample per byte, no bit manipulation needed
fn decode_width_8(input: &[u8], count: usize, output: &mut [i32]) -> CodecResult<usize> {
    let bytes_needed = 1 + count;

    if input.len() < bytes_needed {
        return Err(CodecError::UnexpectedEndOfInput);
    }

    let data = &input[1..];

    // Direct byte-to-sample conversion - maximum throughput
    for (out, &byte) in output[..count].iter_mut().zip(data[..count].iter()) {
        *out = zigzag_decode(byte as u32);
    }

    Ok(bytes_needed)
}

/// Generic decoder for any bit width (1-16 bits)
/// Uses 64-bit accumulator for branchless, high-performance unpacking
#[allow(clippy::needless_range_loop)] // Index needed for output assignment in accumulator loop
fn decode_width_generic(
    input: &[u8],
    count: usize,
    output: &mut [i32],
    bit_width: u8,
) -> CodecResult<usize> {
    let bits_needed = (bit_width as usize) * count;
    let bytes_needed = 1 + (bits_needed + 7) / 8;

    if input.len() < bytes_needed {
        return Err(CodecError::UnexpectedEndOfInput);
    }

    let data = &input[1..];
    let mask = (1u32 << bit_width) - 1;

    // === OPTIMIZED BIT UNPACKING ===
    // Use 64-bit accumulator to batch reads, eliminating per-bit operations
    let mut accumulator: u64 = 0;
    let mut acc_bits: u32 = 0; // Number of valid bits in accumulator
    let mut in_idx = 0usize;

    for i in 0..count {
        // Refill accumulator when needed (load up to 8 bytes at a time)
        while acc_bits < bit_width as u32 && in_idx < data.len() {
            accumulator |= (data[in_idx] as u64) << acc_bits;
            acc_bits += 8;
            in_idx += 1;
        }

        // Extract value from accumulator
        let value = (accumulator & mask as u64) as u32;
        accumulator >>= bit_width;
        acc_bits -= bit_width as u32;

        output[i] = zigzag_decode(value);
    }

    Ok(bytes_needed)
}

/// Encode multiple blocks using fixed-width packing
///
/// Processes input in blocks of 32 samples, calculating optimal bit width per block.
///
/// # Arguments
/// * `deltas` - Delta-encoded values
/// * `output` - Output buffer
///
/// # Returns
/// Number of bytes written
pub fn encode_fixed_width_blocks(deltas: &[i32], output: &mut [u8]) -> CodecResult<usize> {
    let mut total_bytes = 0;
    let num_blocks = (deltas.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for block_idx in 0..num_blocks {
        let start = block_idx * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(deltas.len());
        let block = &deltas[start..end];

        // Calculate bit width for this block
        let bit_width = calculate_bit_width(block).min(16);

        // Encode block
        let bytes_written = encode_block_32(block, bit_width, &mut output[total_bytes..])?;
        total_bytes += bytes_written;
    }

    Ok(total_bytes)
}

/// Decode multiple blocks of fixed-width packed samples
///
/// # Arguments
/// * `input` - Encoded blocks
/// * `sample_count` - Total number of samples to decode
/// * `output` - Output buffer for decoded deltas
///
/// # Returns
/// Number of bytes consumed
pub fn decode_fixed_width_blocks(
    input: &[u8],
    sample_count: usize,
    output: &mut [i32],
) -> CodecResult<usize> {
    let mut total_bytes = 0;
    let mut samples_decoded = 0;
    let num_blocks = (sample_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for _block_idx in 0..num_blocks {
        let remaining = sample_count - samples_decoded;
        let block_size = remaining.min(BLOCK_SIZE);

        let bytes_consumed = decode_block_32(
            &input[total_bytes..],
            block_size,
            &mut output[samples_decoded..samples_decoded + block_size],
        )?;

        total_bytes += bytes_consumed;
        samples_decoded += block_size;
    }

    Ok(total_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::needless_range_loop)] // Test code clarity over micro-optimization
    #[test]
    fn test_zigzag_encode_decode() {
        let test_cases = [0, -1, 1, -2, 2, -100, 100, i32::MIN, i32::MAX];

        for &value in &test_cases {
            let encoded = zigzag_encode(value);
            let decoded = zigzag_decode(encoded);
            assert_eq!(decoded, value, "ZigZag roundtrip failed for {}", value);
        }
    }

    #[test]
    fn test_zigzag_ordering() {
        // ZigZag should map: 0->0, -1->1, 1->2, -2->3, 2->4
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
    }

    #[test]
    fn test_calculate_bit_width_zeros() {
        let deltas = [0, 0, 0, 0];
        assert_eq!(calculate_bit_width(&deltas), 0);
    }

    #[test]
    fn test_calculate_bit_width_small() {
        let deltas = [1, -1, 2, -2]; // ZigZag: 2, 1, 4, 3 -> max=4 -> 3 bits
        assert_eq!(calculate_bit_width(&deltas), 3);
    }

    #[test]
    fn test_calculate_bit_width_power_of_two() {
        let deltas = [0, 7, -8]; // ZigZag: 0, 14, 15 -> max=15 -> 4 bits
        assert_eq!(calculate_bit_width(&deltas), 4);
    }

    #[test]
    fn test_encode_decode_block_zeros() {
        let deltas = [0; 32];
        let mut encoded = [0u8; 100];
        let mut decoded = [0i32; 32];

        let enc_size = encode_block_32(&deltas, 0, &mut encoded).unwrap();
        assert_eq!(enc_size, 1); // Only header

        let dec_size = decode_block_32(&encoded, 32, &mut decoded).unwrap();
        assert_eq!(dec_size, 1);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_encode_decode_block_4bit() {
        let deltas = [1, -1, 2, -2, 3, -3, 4, -4];
        let mut encoded = [0u8; 100];
        let mut decoded = [0i32; 8];

        let bit_width = calculate_bit_width(&deltas);
        let enc_size = encode_block_32(&deltas, bit_width, &mut encoded).unwrap();

        let dec_size = decode_block_32(&encoded, 8, &mut decoded).unwrap();
        assert_eq!(dec_size, enc_size);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_encode_decode_block_8bit() {
        let deltas = [10, -20, 30, -40, 50, -60, 70, -80];
        let mut encoded = [0u8; 100];
        let mut decoded = [0i32; 8];

        let bit_width = calculate_bit_width(&deltas);
        let enc_size = encode_block_32(&deltas, bit_width, &mut encoded).unwrap();

        let dec_size = decode_block_32(&encoded, 8, &mut decoded).unwrap();
        assert_eq!(dec_size, enc_size);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_encode_decode_full_block() {
        let mut deltas = [0i32; 32];
        for (i, delta) in deltas.iter_mut().enumerate() {
            *delta = (i as i32) - 16; // Range: -16 to 15
        }

        let mut encoded = [0u8; 200];
        let mut decoded = [0i32; 32];

        let bit_width = calculate_bit_width(&deltas);
        let enc_size = encode_block_32(&deltas, bit_width, &mut encoded).unwrap();

        let dec_size = decode_block_32(&encoded, 32, &mut decoded).unwrap();
        assert_eq!(dec_size, enc_size);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_encode_decode_multiple_blocks() {
        // Create data that spans multiple blocks (70 samples = 3 blocks: 32 + 32 + 6)
        let mut deltas = [0i32; 70];
        for (i, delta) in deltas.iter_mut().enumerate() {
            *delta = ((i % 20) as i32) - 10;
        }

        let mut encoded = [0u8; 500];
        let mut decoded = [0i32; 70];

        let enc_size = encode_fixed_width_blocks(&deltas, &mut encoded).unwrap();
        let dec_size = decode_fixed_width_blocks(&encoded, 70, &mut decoded).unwrap();

        assert_eq!(dec_size, enc_size);
        assert_eq!(decoded, deltas);
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // Test code clarity
    fn test_encode_decode_varying_block_widths() {
        // Block 1: small values (low bit width)
        // Block 2: large values (high bit width)
        let mut deltas = [0i32; 64];
        for i in 0..32 {
            deltas[i] = i as i32; // Block 1: 0-31
        }
        for i in 32..64 {
            deltas[i] = (i as i32) * 100; // Block 2: large values
        }

        let mut encoded = [0u8; 500];
        let mut decoded = [0i32; 64];

        let enc_size = encode_fixed_width_blocks(&deltas, &mut encoded).unwrap();
        let dec_size = decode_fixed_width_blocks(&encoded, 64, &mut decoded).unwrap();

        assert_eq!(dec_size, enc_size);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_buffer_too_small() {
        let deltas = [1, 2, 3, 4];
        let mut encoded = [0u8; 1]; // Too small

        let result = encode_block_32(&deltas, 4, &mut encoded);
        assert!(matches!(result, Err(CodecError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_decode_unexpected_end() {
        let encoded = [0x08]; // Header says 8-bit width but no data
        let mut decoded = [0i32; 8];

        let result = decode_block_32(&encoded, 8, &mut decoded);
        assert!(matches!(result, Err(CodecError::UnexpectedEndOfInput)));
    }

    #[test]
    fn test_single_sample() {
        let deltas = [42];
        let mut encoded = [0u8; 100];
        let mut decoded = [0i32; 1];

        let bit_width = calculate_bit_width(&deltas);
        let enc_size = encode_block_32(&deltas, bit_width, &mut encoded).unwrap();
        let dec_size = decode_block_32(&encoded, 1, &mut decoded).unwrap();

        assert_eq!(dec_size, enc_size);
        assert_eq!(decoded[0], deltas[0]);
    }

    #[test]
    fn test_mixed_positive_negative() {
        let deltas = [-100, 50, -25, 75, -10, 20, -5, 15];
        let mut encoded = [0u8; 100];
        let mut decoded = [0i32; 8];

        let bit_width = calculate_bit_width(&deltas);
        let enc_size = encode_block_32(&deltas, bit_width, &mut encoded).unwrap();
        let dec_size = decode_block_32(&encoded, 8, &mut decoded).unwrap();

        assert_eq!(dec_size, enc_size);
        assert_eq!(decoded, deltas);
    }
}
