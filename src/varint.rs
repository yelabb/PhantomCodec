//! Varint encoding/decoding implementation
//!
//! Variable-length integer encoding using continuation bits.
//! Each byte stores 7 bits of data; MSB indicates if more bytes follow.

use crate::bitwriter::{BitReader, BitWriter};
use crate::error::{CodecError, CodecResult};

/// Encode array of deltas using varint encoding
///
/// # Arguments
/// * `deltas` - Delta-encoded values
/// * `output` - Output buffer
/// * `use_zigzag` - Whether to apply ZigZag encoding for signed values
///
/// # Returns
/// Number of bytes written
pub fn varint_encode_array(
    deltas: &[i32],
    output: &mut [u8],
    use_zigzag: bool,
) -> CodecResult<usize> {
    let mut writer = BitWriter::new(output);

    for &delta in deltas {
        if use_zigzag {
            writer.write_zigzag(delta)?;
        } else {
            writer.write_varint(delta as u32)?;
        }
    }

    writer.flush()?;
    Ok(writer.bytes_written())
}

/// Decode array of varints
///
/// # Arguments
/// * `input` - Compressed data
/// * `output` - Output buffer for decoded values
/// * `use_zigzag` - Whether to apply ZigZag decoding
///
/// # Returns
/// Number of values decoded
pub fn varint_decode_array(
    input: &[u8],
    output: &mut [i32],
    use_zigzag: bool,
) -> CodecResult<usize> {
    let mut reader = BitReader::new(input);
    let mut count = 0;

    while count < output.len() {
        // Try to read, break on end of input
        let value = if use_zigzag {
            match reader.read_zigzag() {
                Ok(v) => v,
                Err(CodecError::UnexpectedEndOfInput) => break,
                Err(e) => return Err(e),
            }
        } else {
            match reader.read_varint() {
                Ok(v) => v as i32,
                Err(CodecError::UnexpectedEndOfInput) => break,
                Err(e) => return Err(e),
            }
        };

        output[count] = value;
        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encode_decode() {
        let deltas = [10, 3, 2, 3, 2, 1, 1, 0];
        let mut buffer = [0u8; 100];

        let size = varint_encode_array(&deltas, &mut buffer, false).unwrap();
        assert!(size > 0);

        let mut decoded = [0i32; 8];
        let count = varint_decode_array(&buffer[..size], &mut decoded, false).unwrap();
        assert_eq!(count, 8);
        assert_eq!(decoded, deltas);
    }

    #[test]
    fn test_varint_with_zigzag() {
        let deltas = [10, 3, -2, 3, -2, 1, -1, 0];
        let mut buffer = [0u8; 100];

        let size = varint_encode_array(&deltas, &mut buffer, true).unwrap();

        let mut decoded = [0i32; 8];
        let count = varint_decode_array(&buffer[..size], &mut decoded, true).unwrap();
        assert_eq!(count, 8);
        assert_eq!(decoded, deltas);
    }
}
