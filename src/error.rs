//! Error types for PhantomCodec
//!
//! All errors are `Copy` and contain minimal data to ensure they can be used
//! in `no_std` environments without allocation. Error details are encoded
//! as discriminants and primitive values.

use core::fmt;

/// Result type alias for PhantomCodec operations
pub type CodecResult<T> = Result<T, CodecError>;

/// Error types returned by compression/decompression operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// Output buffer is too small for the compressed/decompressed data
    BufferTooSmall {
        /// Number of bytes required
        required: usize,
    },

    /// Input channel count doesn't match expected value
    InvalidChannelCount {
        /// Expected number of channels
        expected: usize,
        /// Actual number of channels found
        actual: usize,
    },

    /// Compressed data header is corrupted (wrong magic bytes)
    CorruptedHeader,

    /// Unsupported codec version
    UnsupportedVersion {
        /// Version found in header
        version: u8,
    },

    /// Invalid strategy ID in packet header
    InvalidStrategy {
        /// Strategy ID that was encountered
        strategy_id: u8,
    },

    /// Compressed data is truncated or corrupted
    UnexpectedEndOfInput,

    /// ZigZag decode encountered invalid bit pattern
    InvalidZigZagValue,

    /// Rice parameter k is out of valid range (0-3)
    InvalidRiceParameter {
        /// The invalid k value
        k: u8,
    },

    /// Rice encoding quotient exceeds maximum safe value
    /// 
    /// This occurs when delta values are too large for the selected Rice parameter.
    /// The value would be silently truncated if encoded, causing data corruption.
    /// Consider using a different compression strategy for this data.
    RiceQuotientOverflow {
        /// The value that exceeded the limit
        value: u32,
        /// The Rice parameter k that was being used
        k: u8,
    },

    /// Bit position overflow in BitWriter
    BitPositionOverflow,
}

impl CodecError {
    /// Convert error to a numeric error code for FFI boundaries
    ///
    /// This allows passing errors across C FFI without allocation.
    /// Error codes are stable and documented.
    pub const fn to_error_code(self) -> i32 {
        match self {
            CodecError::BufferTooSmall { .. } => -1,
            CodecError::InvalidChannelCount { .. } => -2,
            CodecError::CorruptedHeader => -3,
            CodecError::UnsupportedVersion { .. } => -4,
            CodecError::InvalidStrategy { .. } => -5,
            CodecError::UnexpectedEndOfInput => -6,
            CodecError::InvalidZigZagValue => -7,
            CodecError::InvalidRiceParameter { .. } => -8,
            CodecError::RiceQuotientOverflow { .. } => -10,
            CodecError::BitPositionOverflow => -9,
        }
    }

    /// Create error from error code (for FFI boundaries)
    pub const fn from_error_code(code: i32) -> Option<Self> {
        match code {
            -1 => Some(CodecError::BufferTooSmall { required: 0 }),
            -2 => Some(CodecError::InvalidChannelCount {
                expected: 0,
                actual: 0,
            }),
            -3 => Some(CodecError::CorruptedHeader),
            -4 => Some(CodecError::UnsupportedVersion { version: 0 }),
            -5 => Some(CodecError::InvalidStrategy { strategy_id: 0 }),
            -6 => Some(CodecError::UnexpectedEndOfInput),
            -7 => Some(CodecError::InvalidZigZagValue),
            -8 => Some(CodecError::InvalidRiceParameter { k: 0 }),
            -9 => Some(CodecError::BitPositionOverflow),
            -10 => Some(CodecError::RiceQuotientOverflow { value: 0, k: 0 }),
            _ => None,
        }
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::BufferTooSmall { required } => {
                write!(f, "Buffer too small: {} bytes required", required)
            }
            CodecError::InvalidChannelCount { expected, actual } => {
                write!(
                    f,
                    "Invalid channel count: expected {}, got {}",
                    expected, actual
                )
            }
            CodecError::CorruptedHeader => {
                write!(f, "Corrupted packet header (invalid magic bytes)")
            }
            CodecError::UnsupportedVersion { version } => {
                write!(f, "Unsupported codec version: {}", version)
            }
            CodecError::InvalidStrategy { strategy_id } => {
                write!(f, "Invalid compression strategy ID: {}", strategy_id)
            }
            CodecError::UnexpectedEndOfInput => {
                write!(f, "Unexpected end of input (truncated data)")
            }
            CodecError::InvalidZigZagValue => {
                write!(f, "Invalid ZigZag encoded value")
            }
            CodecError::InvalidRiceParameter { k } => {
                write!(f, "Invalid Rice parameter k={} (must be 0-3)", k)
            }
            CodecError::RiceQuotientOverflow { value, k } => {
                write!(
                    f,
                    "Rice quotient overflow: value {} too large for k={} (quotient would exceed 255)",
                    value, k
                )
            }
            CodecError::BitPositionOverflow => {
                write!(f, "Bit position overflow in BitWriter")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_roundtrip() {
        let errors = [
            CodecError::BufferTooSmall { required: 1024 },
            CodecError::InvalidChannelCount {
                expected: 142,
                actual: 128,
            },
            CodecError::CorruptedHeader,
            CodecError::UnsupportedVersion { version: 5 },
            CodecError::InvalidStrategy { strategy_id: 99 },
            CodecError::UnexpectedEndOfInput,
            CodecError::InvalidZigZagValue,
            CodecError::InvalidRiceParameter { k: 7 },
            CodecError::RiceQuotientOverflow { value: 5000, k: 1 },
            CodecError::BitPositionOverflow,
        ];

        for error in &errors {
            let code = error.to_error_code();
            let reconstructed = CodecError::from_error_code(code);
            assert!(reconstructed.is_some());
            // Note: We can't check exact equality because we lose parameter details
            assert_eq!(reconstructed.unwrap().to_error_code(), code);
        }
    }

    #[test]
    fn test_invalid_error_code() {
        assert!(CodecError::from_error_code(0).is_none());
        assert!(CodecError::from_error_code(100).is_none());
        assert!(CodecError::from_error_code(-100).is_none());
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_error_display() {
        let err = CodecError::BufferTooSmall { required: 2048 };
        let display = format!("{}", err);
        assert!(display.contains("2048"));

        let err = CodecError::InvalidChannelCount {
            expected: 1024,
            actual: 512,
        };
        let display = format!("{}", err);
        assert!(display.contains("1024"));
        assert!(display.contains("512"));

        let err = CodecError::RiceQuotientOverflow { value: 5000, k: 1 };
        let display = format!("{}", err);
        assert!(display.contains("5000"));
        assert!(display.contains("k=1"));
    }
}
