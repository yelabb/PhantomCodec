//! Compression strategy trait and implementations
//!
//! Defines the core abstraction for compression algorithms. Strategies use
//! const generics and trait constants to enable compile-time specialization
//! and dead-code elimination via monomorphization.

use crate::error::{CodecError, CodecResult};

/// Magic bytes for PhantomCodec packets: "PHDC" (0x50484443)
pub const MAGIC_BYTES: [u8; 4] = [0x50, 0x48, 0x44, 0x43];

/// Current protocol version
pub const PROTOCOL_VERSION: u8 = 0x01;

/// Strategy IDs encoded in packet header
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StrategyId {
    /// Delta + Varint encoding (for unsigned spike counts)
    DeltaVarint = 0x00,
    /// Adaptive Rice coding (for signed voltages)
    Rice = 0x01,
    /// Fixed 4-bit packing (lossy, ultra-low-latency <10µs)
    Packed4 = 0x02,
    /// Fixed-Width Block Packing (PFOR) - lossless, ultra-low-latency <10µs
    FixedWidth = 0x03,
}

impl StrategyId {
    /// Convert u8 to StrategyId
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(StrategyId::DeltaVarint),
            0x01 => Some(StrategyId::Rice),
            0x02 => Some(StrategyId::Packed4),
            0x03 => Some(StrategyId::FixedWidth),
            _ => None,
        }
    }
}

/// Core compression strategy trait
///
/// Implementors provide compression and decompression logic with compile-time
/// configuration via associated constants. The compiler uses monomorphization
/// to generate specialized code for each strategy, eliminating runtime overhead.
pub trait CompressionStrategy {
    /// Whether this strategy requires ZigZag encoding for signed integers
    ///
    /// When `false`, the compiler will eliminate ZigZag code paths entirely.
    const REQUIRES_ZIGZAG: bool;

    /// Strategy identifier for packet header
    const STRATEGY_ID: StrategyId;

    /// Compress input data into output buffer
    ///
    /// # Arguments
    /// * `input` - Raw neural data (spike counts or voltages)
    /// * `output` - Destination buffer for compressed data
    ///
    /// # Returns
    /// Number of bytes written to output, or error if buffer too small
    fn compress(input: &[i32], output: &mut [u8]) -> CodecResult<usize>;

    /// Decompress input data into output buffer
    ///
    /// # Arguments
    /// * `input` - Compressed data packet
    /// * `output` - Destination buffer for decompressed data
    ///
    /// # Returns
    /// Number of elements decompressed, or error if data corrupted
    fn decompress(input: &[u8], output: &mut [i32]) -> CodecResult<usize>;
}

/// Packet header structure (8 bytes)
///
/// ```text
/// ┌──────────┬──────────┬──────────────────┬────────────────────┐
/// │ Magic    │ Version  │ Channel Count    │ Strategy | Rice k │
/// │ (4 bytes)│ (1 byte) │ (2 bytes BE)     │ (1 byte)           │
/// └──────────┴──────────┴──────────────────┴────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    /// Number of channels in this packet
    pub channel_count: u16,
    /// Compression strategy used
    pub strategy_id: StrategyId,
    /// Rice parameter k (0-3), only used for Rice strategy
    pub rice_k: u8,
}

impl PacketHeader {
    /// Size of header in bytes
    pub const SIZE: usize = 8;

    /// Create new header
    pub const fn new(channel_count: u16, strategy_id: StrategyId, rice_k: u8) -> Self {
        Self {
            channel_count,
            strategy_id,
            rice_k,
        }
    }

    /// Write header to buffer
    ///
    /// # Panics
    /// Panics if buffer is smaller than SIZE (compile-time guaranteed in typical usage)
    pub fn write(&self, buffer: &mut [u8]) -> CodecResult<()> {
        if buffer.len() < Self::SIZE {
            return Err(CodecError::BufferTooSmall {
                required: Self::SIZE,
            });
        }

        // Magic bytes
        buffer[0..4].copy_from_slice(&MAGIC_BYTES);

        // Version
        buffer[4] = PROTOCOL_VERSION;

        // Channel count (big-endian u16)
        buffer[5] = (self.channel_count >> 8) as u8;
        buffer[6] = (self.channel_count & 0xFF) as u8;

        // Strategy ID (6 bits) | Rice k (2 bits)
        let strategy_byte = ((self.strategy_id as u8) << 2) | (self.rice_k & 0x03);
        buffer[7] = strategy_byte;

        Ok(())
    }

    /// Read header from buffer
    pub fn read(buffer: &[u8]) -> CodecResult<Self> {
        if buffer.len() < Self::SIZE {
            return Err(CodecError::UnexpectedEndOfInput);
        }

        // Verify magic bytes
        if buffer[0..4] != MAGIC_BYTES {
            return Err(CodecError::CorruptedHeader);
        }

        // Check version
        let version = buffer[4];
        if version != PROTOCOL_VERSION {
            return Err(CodecError::UnsupportedVersion { version });
        }

        // Parse channel count (big-endian u16)
        let channel_count = ((buffer[5] as u16) << 8) | (buffer[6] as u16);

        // Parse strategy byte
        let strategy_byte = buffer[7];
        let strategy_id_raw = strategy_byte >> 2;
        let rice_k = strategy_byte & 0x03;

        let strategy_id =
            StrategyId::from_u8(strategy_id_raw).ok_or(CodecError::InvalidStrategy {
                strategy_id: strategy_id_raw,
            })?;

        Ok(Self {
            channel_count,
            strategy_id,
            rice_k,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header_roundtrip() {
        let header = PacketHeader::new(1024, StrategyId::DeltaVarint, 0);
        let mut buffer = [0u8; 8];

        header.write(&mut buffer).unwrap();
        let decoded = PacketHeader::read(&buffer).unwrap();

        assert_eq!(decoded.channel_count, 1024);
        assert_eq!(decoded.strategy_id, StrategyId::DeltaVarint);
        assert_eq!(decoded.rice_k, 0);
    }

    #[test]
    fn test_packet_header_with_rice() {
        let header = PacketHeader::new(512, StrategyId::Rice, 3);
        let mut buffer = [0u8; 8];

        header.write(&mut buffer).unwrap();
        let decoded = PacketHeader::read(&buffer).unwrap();

        assert_eq!(decoded.channel_count, 512);
        assert_eq!(decoded.strategy_id, StrategyId::Rice);
        assert_eq!(decoded.rice_k, 3);
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let mut buffer = [0u8; 8];
        buffer[0] = 0xFF; // Wrong magic

        let result = PacketHeader::read(&buffer);
        assert_eq!(result, Err(CodecError::CorruptedHeader));
    }

    #[test]
    fn test_unsupported_version() {
        let mut buffer = [0u8; 8];
        buffer[0..4].copy_from_slice(&MAGIC_BYTES);
        buffer[4] = 0xFF; // Unsupported version

        let result = PacketHeader::read(&buffer);
        assert!(matches!(result, Err(CodecError::UnsupportedVersion { .. })));
    }

    #[test]
    fn test_invalid_strategy() {
        let mut buffer = [0u8; 8];
        buffer[0..4].copy_from_slice(&MAGIC_BYTES);
        buffer[4] = PROTOCOL_VERSION;
        buffer[5] = 0;
        buffer[6] = 64; // 64 channels
        buffer[7] = 0xFF; // Invalid strategy (top 6 bits)

        let result = PacketHeader::read(&buffer);
        assert!(matches!(result, Err(CodecError::InvalidStrategy { .. })));
    }

    #[test]
    fn test_buffer_too_small() {
        let header = PacketHeader::new(128, StrategyId::DeltaVarint, 0);
        let mut buffer = [0u8; 4]; // Too small

        let result = header.write(&mut buffer);
        assert!(matches!(result, Err(CodecError::BufferTooSmall { .. })));
    }
}
