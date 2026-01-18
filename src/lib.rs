//! PhantomCodec - Real-time lossless compression for high-density neural data
//!
//! A `#![no_std]` compatible crate for compressing 1,024+ channel neural spike data
//! with <10μs decode latency and zero allocations in the hot path.
//!
//! # Compiler Requirements
//!
//! - **Stable Rust**: Works with scalar fallback (no `simd` feature)
//! - **Nightly Rust**: Required for `simd` feature (uses unstable `core::simd`)
//!
//! # Example
//! ```
//! use phantomcodec::{compress_spike_counts, decompress_spike_counts};
//!
//! // Simulate neural data (1024 channels)
//! let mut spike_counts = [0i32; 1024];
//! spike_counts[42] = 7;  // Channel 42 fired 7 times
//! spike_counts[99] = 3;  // Channel 99 fired 3 times
//!
//! // Allocate workspace (required for no_std safety)
//! let mut workspace = [0i32; 1024];
//!
//! // Compress
//! let mut compressed = [0u8; 8192];
//! let size = compress_spike_counts(&spike_counts, &mut compressed, &mut workspace)
//!     .expect("Compression failed");
//!
//! // Decompress
//! let mut decompressed = [0i32; 1024];
//! decompress_spike_counts(&compressed[..size], &mut decompressed, &mut workspace)
//!     .expect("Decompression failed");
//!
//! assert_eq!(spike_counts, decompressed);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(all(feature = "simd", target_feature = "simd128"), feature(portable_simd))]
#![warn(missing_docs)]

pub mod bitwriter;
pub mod buffer;
pub mod error;
pub mod rice;
pub mod simd;
pub mod strategy;

mod varint;

// Re-export commonly used types
pub use error::{CodecError, CodecResult};
pub use strategy::{CompressionStrategy, PacketHeader, StrategyId};

use buffer::NeuralFrame;
use simd::{compute_deltas, reconstruct_from_deltas};
use varint::{varint_decode_array, varint_encode_array};

/// High-level API: Compress spike counts using Delta + Varint encoding
///
/// Optimized for unsigned spike count data (no ZigZag encoding needed).
/// Uses SIMD-accelerated delta computation when available.
///
/// # Arguments
/// * `input` - Raw spike counts per channel
/// * `output` - Output buffer for compressed packet
/// * `workspace` - Temporary buffer for delta computation (must be >= input.len())
///
/// # Returns
/// Number of bytes written, or error if buffer too small
///
/// # Safety
/// The `workspace` buffer is required for no_std environments to avoid
/// unsafe static mutable state. Caller must ensure workspace is not aliased.
///
/// # Example
/// ```
/// # use phantomcodec::compress_spike_counts;
/// let spike_counts = [1, 4, 2, 5, 3];
/// let mut compressed = [0u8; 100];
/// let mut workspace = [0i32; 5];
/// let size = compress_spike_counts(&spike_counts, &mut compressed, &mut workspace).unwrap();
/// assert!(size > 0);
/// ```
pub fn compress_spike_counts(input: &[i32], output: &mut [u8], workspace: &mut [i32]) -> CodecResult<usize> {
    let frame = NeuralFrame::new(input);
    let channel_count = frame.channel_count_u16()?;

    // Write packet header
    let header = PacketHeader::new(channel_count, StrategyId::DeltaVarint, 0);
    header.write(output)?;

    let payload_start = PacketHeader::SIZE;

    // Validate workspace size
    if workspace.len() < input.len() {
        return Err(CodecError::BufferTooSmall {
            required: input.len(),
        });
    }

    // Compute deltas using SIMD
    compute_deltas(input, &mut workspace[..input.len()]);

    // Encode with Varint (no ZigZag for unsigned data)
    let payload_size = varint_encode_array(
        &workspace[..input.len()],
        &mut output[payload_start..],
        false, // No ZigZag
    )?;

    Ok(payload_start + payload_size)
}

/// High-level API: Decompress spike counts
///
/// # Arguments
/// * `input` - Compressed packet (including header)
/// * `output` - Output buffer for decompressed spike counts
/// * `workspace` - Temporary buffer for delta reconstruction (must be >= channel_count)
///
/// # Returns
/// Number of channels decompressed
///
/// # Safety
/// The `workspace` buffer is required for no_std environments to avoid
/// unsafe static mutable state. Caller must ensure workspace is not aliased.
///
/// # Example
/// ```
/// # use phantomcodec::{compress_spike_counts, decompress_spike_counts};
/// let original = [1, 4, 2, 5, 3];
/// let mut compressed = [0u8; 100];
/// let mut workspace = [0i32; 5];
/// let size = compress_spike_counts(&original, &mut compressed, &mut workspace).unwrap();
///
/// let mut decompressed = [0i32; 5];
/// let count = decompress_spike_counts(&compressed[..size], &mut decompressed, &mut workspace).unwrap();
/// assert_eq!(count, 5);
/// assert_eq!(original, decompressed);
/// ```
pub fn decompress_spike_counts(input: &[u8], output: &mut [i32], workspace: &mut [i32]) -> CodecResult<usize> {
    // Parse header
    let header = PacketHeader::read(input)?;

    if header.strategy_id != StrategyId::DeltaVarint {
        return Err(CodecError::InvalidStrategy {
            strategy_id: header.strategy_id as u8,
        });
    }

    let channel_count = header.channel_count as usize;
    if output.len() < channel_count {
        return Err(CodecError::BufferTooSmall {
            required: channel_count,
        });
    }

    // Validate workspace size
    if workspace.len() < channel_count {
        return Err(CodecError::BufferTooSmall {
            required: channel_count,
        });
    }

    // Decode deltas
    let payload = &input[PacketHeader::SIZE..];
    varint_decode_array(payload, &mut workspace[..channel_count], false)?;

    // Reconstruct from deltas
    reconstruct_from_deltas(&workspace[..channel_count], &mut output[..channel_count]);

    Ok(channel_count)
}

/// High-level API: Compress voltage data using adaptive Rice coding
///
/// Optimized for signed voltage samples with ZigZag encoding.
/// Automatically selects Rice parameter k based on data characteristics.
///
/// # Arguments
/// * `input` - Raw voltage samples
/// * `output` - Output buffer for compressed packet
/// * `workspace` - Temporary buffer for delta computation (must be >= input.len())
///
/// # Returns
/// Number of bytes written, or error if buffer too small
///
/// # Safety
/// The `workspace` buffer is required for no_std environments to avoid
/// unsafe static mutable state. Caller must ensure workspace is not aliased.
pub fn compress_voltage(input: &[i32], output: &mut [u8], workspace: &mut [i32]) -> CodecResult<usize> {
    let frame = NeuralFrame::new(input);
    let channel_count = frame.channel_count_u16()?;

    // Validate workspace size
    if workspace.len() < input.len() {
        return Err(CodecError::BufferTooSmall {
            required: input.len(),
        });
    }

    // Compute deltas
    compute_deltas(input, &mut workspace[..input.len()]);

    // Encode with Rice (with ZigZag for signed data)
    let (payload_size, k) = rice::rice_encode_array(
        &workspace[..input.len()],
        &mut output[PacketHeader::SIZE..],
        true, // Use ZigZag
    )?;

    // Write header with selected k
    let header = PacketHeader::new(channel_count, StrategyId::Rice, k);
    header.write(output)?;

    Ok(PacketHeader::SIZE + payload_size)
}

/// High-level API: Decompress voltage data
///
/// # Arguments
/// * `input` - Compressed packet (including header)
/// * `output` - Output buffer for decompressed voltages
/// * `workspace` - Temporary buffer for delta reconstruction (must be >= channel_count)
///
/// # Returns
/// Number of samples decompressed
///
/// # Safety
/// The `workspace` buffer is required for no_std environments to avoid
/// unsafe static mutable state. Caller must ensure workspace is not aliased.
pub fn decompress_voltage(input: &[u8], output: &mut [i32], workspace: &mut [i32]) -> CodecResult<usize> {
    // Parse header
    let header = PacketHeader::read(input)?;

    if header.strategy_id != StrategyId::Rice {
        return Err(CodecError::InvalidStrategy {
            strategy_id: header.strategy_id as u8,
        });
    }

    let channel_count = header.channel_count as usize;
    if output.len() < channel_count {
        return Err(CodecError::BufferTooSmall {
            required: channel_count,
        });
    }

    // Validate workspace size
    if workspace.len() < channel_count {
        return Err(CodecError::BufferTooSmall {
            required: channel_count,
        });
    }

    // Decode deltas
    let payload = &input[PacketHeader::SIZE..];
    rice::rice_decode_array(payload, &mut workspace[..channel_count], header.rice_k, true)?;

    // Reconstruct from deltas
    reconstruct_from_deltas(&workspace[..channel_count], &mut output[..channel_count]);

    Ok(channel_count)
}

/// Generic compression with custom strategy
///
/// For advanced users who want explicit control over the compression algorithm.
///
/// # Type Parameters
/// * `S` - Compression strategy (must implement `CompressionStrategy`)
///
/// # Example
/// ```ignore
/// use phantomcodec::{compress, DeltaVarintStrategy};
/// let size = compress::<DeltaVarintStrategy>(&input, &mut output)?;
/// ```
pub fn compress<S: CompressionStrategy>(input: &[i32], output: &mut [u8]) -> CodecResult<usize> {
    S::compress(input, output)
}

/// Generic decompression with custom strategy
///
/// # Type Parameters
/// * `S` - Compression strategy (must match the one used for compression)
pub fn decompress<S: CompressionStrategy>(
    input: &[u8],
    output: &mut [i32],
) -> CodecResult<usize> {
    S::decompress(input, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_counts_roundtrip() {
        let original = [1, 4, 2, 5, 3, 6, 2, 7, 1, 8];
        let mut compressed = [0u8; 100];
        let mut decompressed = [0i32; 10];
        let mut workspace = [0i32; 10];

        let size = compress_spike_counts(&original, &mut compressed, &mut workspace).unwrap();
        assert!(size > 0);
        assert!(size < 100); // Should be compressed

        let count = decompress_spike_counts(&compressed[..size], &mut decompressed, &mut workspace).unwrap();
        assert_eq!(count, 10);
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_voltage_roundtrip() {
        let original = [100, 103, 101, 104, 102, 105, 103, 106, 104, 107];
        let mut compressed = [0u8; 100];
        let mut decompressed = [0i32; 10];
        let mut workspace = [0i32; 10];

        let size = compress_voltage(&original, &mut compressed, &mut workspace).unwrap();
        assert!(size > 0);

        let count = decompress_voltage(&compressed[..size], &mut decompressed, &mut workspace).unwrap();
        assert_eq!(count, 10);
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_sparse_data() {
        // Mostly zeros (typical neural data)
        let mut original = [0i32; 100];
        original[10] = 3;
        original[50] = 7;
        original[90] = 2;

        let mut compressed = [0u8; 400];
        let mut decompressed = [0i32; 100];
        let mut workspace = [0i32; 100];

        let size = compress_spike_counts(&original, &mut compressed, &mut workspace).unwrap();
        decompress_spike_counts(&compressed[..size], &mut decompressed, &mut workspace).unwrap();

        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_buffer_too_small() {
        let original = [1, 2, 3, 4, 5];
        let mut compressed = [0u8; 5]; // Too small
        let mut workspace = [0i32; 5];

        let result = compress_spike_counts(&original, &mut compressed, &mut workspace);
        assert!(matches!(result, Err(CodecError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_corrupted_header() {
        let corrupted = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut output = [0i32; 10];
        let mut workspace = [0i32; 10];

        let result = decompress_spike_counts(&corrupted, &mut output, &mut workspace);
        assert_eq!(result, Err(CodecError::CorruptedHeader));
    }

    #[test]
    fn test_realistic_size() {
        // Simulate 142 channels @ 40Hz (typical PhantomLink data)
        let mut original = [0i32; 142];
        for i in 0..142 {
            original[i] = (i % 5) as i32; // Sparse firing
        }

        let mut compressed = [0u8; 1024];
        let mut workspace = [0i32; 142];
        let size = compress_spike_counts(&original, &mut compressed, &mut workspace).unwrap();

        // Expect significant compression
        println!("Original: {} bytes, Compressed: {} bytes", 142 * 4, size);
        assert!(size < 142 * 4); // Should be smaller than raw data
    }
}
