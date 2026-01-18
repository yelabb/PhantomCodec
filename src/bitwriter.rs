//! Bit-level writing abstractions for variable-length encoding
//!
//! Provides safe bit-packing operations without unsafe pointer arithmetic.
//! All operations maintain strict bounds checking and return errors on overflow.

use crate::error::{CodecError, CodecResult};

/// Bit-level writer for packing variable-width values into byte buffers
///
/// Maintains a bit-position cursor and handles byte-boundary alignment
/// automatically. All operations are safe and bounds-checked.
///
/// # Example
/// ```
/// # use phantomcodec::bitwriter::BitWriter;
/// let mut buffer = [0u8; 10];
/// let mut writer = BitWriter::new(&mut buffer);
///
/// writer.write_bits(0b1010, 4).unwrap();  // Write 4 bits
/// writer.write_bits(0b11, 2).unwrap();    // Write 2 more bits
/// writer.flush().unwrap();
///
/// assert_eq!(buffer[0], 0b10101100); // Bits written MSB first
/// ```
pub struct BitWriter<'a> {
    buffer: &'a mut [u8],
    /// Current bit position (0 = first bit of first byte)
    bit_pos: usize,
}

impl<'a> BitWriter<'a> {
    /// Create a new BitWriter wrapping the given buffer
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self { buffer, bit_pos: 0 }
    }

    /// Get current bit position
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Get number of complete bytes written
    pub fn bytes_written(&self) -> usize {
        (self.bit_pos + 7) / 8
    }

    /// Get buffer capacity in bits
    pub fn capacity_bits(&self) -> usize {
        self.buffer.len() * 8
    }

    /// Get remaining capacity in bits
    pub fn remaining_bits(&self) -> usize {
        self.capacity_bits().saturating_sub(self.bit_pos)
    }

    /// Write `width` bits from `value` (LSB-aligned)
    ///
    /// # Arguments
    /// * `value` - Value to write (only lowest `width` bits are used)
    /// * `width` - Number of bits to write (1-32)
    ///
    /// # Errors
    /// Returns `BufferTooSmall` if not enough space remains
    ///
    /// # Example
    /// ```
    /// # use phantomcodec::bitwriter::BitWriter;
    /// let mut buffer = [0u8; 2];
    /// let mut writer = BitWriter::new(&mut buffer);
    ///
    /// writer.write_bits(0b1111, 4).unwrap();
    /// writer.write_bits(0b10, 2).unwrap();
    /// writer.flush().unwrap();
    ///
    /// assert_eq!(buffer[0], 0b11111000);
    /// ```
    pub fn write_bits(&mut self, value: u32, width: u8) -> CodecResult<()> {
        if width == 0 || width > 32 {
            return Err(CodecError::BitPositionOverflow);
        }

        if self.remaining_bits() < width as usize {
            return Err(CodecError::BufferTooSmall {
                required: self.bytes_written() + ((width as usize + 7) / 8),
            });
        }

        // Mask value to width bits
        let mask = if width == 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let value = value & mask;

        // Write bits MSB first
        for i in (0..width).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_idx = self.bit_pos / 8;
            let bit_idx = self.bit_pos % 8;

            if byte_idx >= self.buffer.len() {
                return Err(CodecError::BitPositionOverflow);
            }

            if bit != 0 {
                self.buffer[byte_idx] |= 1 << (7 - bit_idx);
            }

            self.bit_pos += 1;
        }

        Ok(())
    }

    /// Write a varint-encoded unsigned integer
    ///
    /// Uses continuation-bit encoding: MSB=1 means more bytes follow.
    /// Each byte stores 7 bits of data.
    ///
    /// # Example
    /// ```
    /// # use phantomcodec::bitwriter::BitWriter;
    /// let mut buffer = [0u8; 10];
    /// let mut writer = BitWriter::new(&mut buffer);
    ///
    /// writer.write_varint(127).unwrap();  // Fits in 1 byte
    /// writer.write_varint(300).unwrap();  // Requires 2 bytes
    /// ```
    pub fn write_varint(&mut self, mut value: u32) -> CodecResult<()> {
        // Align to byte boundary first (varint must be byte-aligned)
        self.flush()?;

        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80; // Set continuation bit
            }

            let byte_idx = self.bit_pos / 8;
            if byte_idx >= self.buffer.len() {
                return Err(CodecError::BufferTooSmall {
                    required: byte_idx + 1,
                });
            }

            self.buffer[byte_idx] = byte;
            self.bit_pos += 8;

            if value == 0 {
                break;
            }
        }

        Ok(())
    }

    /// Write a zigzag-encoded signed integer
    ///
    /// ZigZag mapping: (n << 1) ^ (n >> 31)
    /// Maps: 0, -1, 1, -2, 2, ... → 0, 1, 2, 3, 4, ...
    ///
    /// # Example
    /// ```
    /// # use phantomcodec::bitwriter::BitWriter;
    /// let mut buffer = [0u8; 10];
    /// let mut writer = BitWriter::new(&mut buffer);
    ///
    /// writer.write_zigzag(-1).unwrap();  // Encoded as 1
    /// writer.write_zigzag(1).unwrap();   // Encoded as 2
    /// writer.write_zigzag(-2).unwrap();  // Encoded as 3
    /// ```
    pub fn write_zigzag(&mut self, value: i32) -> CodecResult<()> {
        let encoded = ((value << 1) ^ (value >> 31)) as u32;
        self.write_varint(encoded)
    }

    /// Flush partial byte by padding with zeros
    ///
    /// Advances to next byte boundary if not already aligned.
    pub fn flush(&mut self) -> CodecResult<()> {
        let remainder = self.bit_pos % 8;
        if remainder != 0 {
            let padding = 8 - remainder;
            self.bit_pos += padding;
        }
        Ok(())
    }

    /// Reset writer to beginning of buffer
    pub fn reset(&mut self) {
        self.bit_pos = 0;
        self.buffer.fill(0);
    }
}

/// Bit-level reader for unpacking variable-width values
pub struct BitReader<'a> {
    buffer: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a new BitReader
    pub fn new(buffer: &'a [u8]) -> Self {
        Self { buffer, bit_pos: 0 }
    }

    /// Get current bit position
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Get remaining bits
    pub fn remaining_bits(&self) -> usize {
        (self.buffer.len() * 8).saturating_sub(self.bit_pos)
    }

    /// Read `width` bits into a u32
    pub fn read_bits(&mut self, width: u8) -> CodecResult<u32> {
        if width == 0 || width > 32 {
            return Err(CodecError::BitPositionOverflow);
        }

        if self.remaining_bits() < width as usize {
            return Err(CodecError::UnexpectedEndOfInput);
        }

        let mut value = 0u32;

        for _ in 0..width {
            let byte_idx = self.bit_pos / 8;
            let bit_idx = self.bit_pos % 8;

            if byte_idx >= self.buffer.len() {
                return Err(CodecError::UnexpectedEndOfInput);
            }

            let bit = (self.buffer[byte_idx] >> (7 - bit_idx)) & 1;
            value = (value << 1) | (bit as u32);

            self.bit_pos += 1;
        }

        Ok(value)
    }

    /// Read a varint-encoded unsigned integer
    pub fn read_varint(&mut self) -> CodecResult<u32> {
        // Align to byte boundary
        let remainder = self.bit_pos % 8;
        if remainder != 0 {
            self.bit_pos += 8 - remainder;
        }

        let mut value = 0u32;
        let mut shift = 0u32;

        loop {
            let byte_idx = self.bit_pos / 8;
            if byte_idx >= self.buffer.len() {
                return Err(CodecError::UnexpectedEndOfInput);
            }

            let byte = self.buffer[byte_idx];
            self.bit_pos += 8;

            value |= ((byte & 0x7F) as u32) << shift;
            shift += 7;

            if shift > 35 {
                // Prevent overflow (max 5 bytes for u32 = 35 bits)
                return Err(CodecError::BitPositionOverflow);
            }

            if (byte & 0x80) == 0 {
                break;
            }
        }

        Ok(value)
    }

    /// Read a zigzag-encoded signed integer
    pub fn read_zigzag(&mut self) -> CodecResult<i32> {
        let encoded = self.read_varint()?;
        let decoded = ((encoded >> 1) as i32) ^ (-((encoded & 1) as i32));
        Ok(decoded)
    }

    /// Skip to next byte boundary
    pub fn align(&mut self) {
        let remainder = self.bit_pos % 8;
        if remainder != 0 {
            self.bit_pos += 8 - remainder;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_bits() {
        let mut buffer = [0u8; 2];
        let mut writer = BitWriter::new(&mut buffer);

        writer.write_bits(0b1010, 4).unwrap();
        writer.write_bits(0b11, 2).unwrap();
        writer.flush().unwrap();

        assert_eq!(buffer[0], 0b10101100);
    }

    #[test]
    fn test_varint_roundtrip() {
        let mut buffer = [0u8; 10];
        let mut writer = BitWriter::new(&mut buffer);

        writer.write_varint(0).unwrap();
        writer.write_varint(127).unwrap();
        writer.write_varint(128).unwrap();
        writer.write_varint(16383).unwrap();

        let mut reader = BitReader::new(&buffer);
        assert_eq!(reader.read_varint().unwrap(), 0);
        assert_eq!(reader.read_varint().unwrap(), 127);
        assert_eq!(reader.read_varint().unwrap(), 128);
        assert_eq!(reader.read_varint().unwrap(), 16383);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        let mut buffer = [0u8; 10];
        let mut writer = BitWriter::new(&mut buffer);

        let values = [0, -1, 1, -2, 2, -64, 64];
        for &val in &values {
            writer.write_zigzag(val).unwrap();
        }

        let mut reader = BitReader::new(&buffer);
        for &expected in &values {
            assert_eq!(reader.read_zigzag().unwrap(), expected);
        }
    }

    #[test]
    fn test_bits_roundtrip() {
        let mut buffer = [0u8; 10];
        let mut writer = BitWriter::new(&mut buffer);

        writer.write_bits(0b1111, 4).unwrap();
        writer.write_bits(0b10, 2).unwrap();
        writer.write_bits(0b101010, 6).unwrap();
        writer.flush().unwrap();

        let mut reader = BitReader::new(&buffer);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1111);
        assert_eq!(reader.read_bits(2).unwrap(), 0b10);
        assert_eq!(reader.read_bits(6).unwrap(), 0b101010);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buffer = [0u8; 1];
        let mut writer = BitWriter::new(&mut buffer);

        writer.write_bits(0xFF, 8).unwrap();
        let result = writer.write_bits(0xFF, 8);
        assert!(matches!(result, Err(CodecError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_reader_underflow() {
        let buffer = [0xFF];
        let mut reader = BitReader::new(&buffer);

        reader.read_bits(8).unwrap();
        let result = reader.read_bits(8);
        assert_eq!(result, Err(CodecError::UnexpectedEndOfInput));
    }
}
