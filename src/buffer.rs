//! Zero-copy buffer abstractions for neural data
//!
//! Provides lifetime-bound wrappers around raw slices that enable safe
//! in-place transformations without allocation. All memory is provided
//! by the application layer (stack, static, or DMA regions).

use crate::error::{CodecError, CodecResult};

/// Wrapper around raw neural data (spike counts or voltages)
///
/// This is a zero-copy view into application-provided memory.
/// The lifetime `'a` ensures the underlying data remains valid.
#[derive(Debug)]
pub struct NeuralFrame<'a> {
    data: &'a [i32],
}

impl<'a> NeuralFrame<'a> {
    /// Create a new neural frame from a slice
    ///
    /// # Arguments
    /// * `data` - Raw neural data (spike counts or voltage samples)
    ///
    /// # Example
    /// ```
    /// # use phantomcodec::buffer::NeuralFrame;
    /// let data = [1, 2, 3, 4, 5];
    /// let frame = NeuralFrame::new(&data);
    /// assert_eq!(frame.len(), 5);
    /// ```
    pub fn new(data: &'a [i32]) -> Self {
        Self { data }
    }

    /// Get number of channels/samples in frame
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if frame is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get underlying data slice
    pub fn as_slice(&self) -> &[i32] {
        self.data
    }

    /// Get channel count as u16 (for packet headers)
    ///
    /// Returns error if channel count exceeds u16::MAX
    pub fn channel_count_u16(&self) -> CodecResult<u16> {
        self.data
            .len()
            .try_into()
            .map_err(|_| CodecError::InvalidChannelCount {
                expected: u16::MAX as usize,
                actual: self.data.len(),
            })
    }
}

/// Wrapper around compressed packet buffer
///
/// Manages a mutable byte buffer for writing compressed data.
/// The lifetime `'a` ensures safe borrowing without allocation.
#[derive(Debug)]
pub struct CompressedPacket<'a> {
    buffer: &'a mut [u8],
    bytes_written: usize,
}

impl<'a> CompressedPacket<'a> {
    /// Create a new compressed packet buffer
    ///
    /// # Arguments
    /// * `buffer` - Mutable slice for compressed output
    ///
    /// # Example
    /// ```
    /// # use phantomcodec::buffer::CompressedPacket;
    /// let mut buffer = [0u8; 1024];
    /// let packet = CompressedPacket::new(&mut buffer);
    /// assert_eq!(packet.capacity(), 1024);
    /// ```
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            buffer,
            bytes_written: 0,
        }
    }

    /// Get total capacity of buffer
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get number of bytes written so far
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }

    /// Get remaining capacity
    pub fn remaining(&self) -> usize {
        self.buffer.len().saturating_sub(self.bytes_written)
    }

    /// Get slice of written data
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer[..self.bytes_written]
    }

    /// Get mutable slice of entire buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer
    }

    /// Get mutable slice of unwritten portion
    pub fn remaining_mut(&mut self) -> &mut [u8] {
        &mut self.buffer[self.bytes_written..]
    }

    /// Advance write cursor by `n` bytes
    ///
    /// # Safety
    /// Caller must ensure that `n` bytes have actually been written
    pub fn advance(&mut self, n: usize) -> CodecResult<()> {
        if self.bytes_written + n > self.buffer.len() {
            return Err(CodecError::BufferTooSmall {
                required: self.bytes_written + n,
            });
        }
        self.bytes_written += n;
        Ok(())
    }

    /// Reset write cursor to beginning
    pub fn reset(&mut self) {
        self.bytes_written = 0;
    }

    /// Split buffer at current write position
    ///
    /// Returns (written_data, remaining_buffer)
    pub fn split(&mut self) -> (&[u8], &mut [u8]) {
        let (written, remaining) = self.buffer.split_at_mut(self.bytes_written);
        (written, remaining)
    }
}

/// Wrapper around decompression input buffer
///
/// Provides safe reading from compressed packet with bounds checking.
#[derive(Debug)]
pub struct CompressedInput<'a> {
    data: &'a [u8],
    read_pos: usize,
}

impl<'a> CompressedInput<'a> {
    /// Create new input buffer
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, read_pos: 0 }
    }

    /// Get total length of input
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if input is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get current read position
    pub fn position(&self) -> usize {
        self.read_pos
    }

    /// Get remaining unread bytes
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.read_pos)
    }

    /// Get slice of remaining data
    pub fn remaining_slice(&self) -> &[u8] {
        &self.data[self.read_pos..]
    }

    /// Read exactly `n` bytes
    ///
    /// Returns error if not enough data remaining
    pub fn read_bytes(&mut self, n: usize) -> CodecResult<&[u8]> {
        if self.remaining() < n {
            return Err(CodecError::UnexpectedEndOfInput);
        }
        let slice = &self.data[self.read_pos..self.read_pos + n];
        self.read_pos += n;
        Ok(slice)
    }

    /// Read a single byte
    pub fn read_u8(&mut self) -> CodecResult<u8> {
        if self.remaining() < 1 {
            return Err(CodecError::UnexpectedEndOfInput);
        }
        let byte = self.data[self.read_pos];
        self.read_pos += 1;
        Ok(byte)
    }

    /// Peek at next byte without advancing cursor
    pub fn peek_u8(&self) -> CodecResult<u8> {
        if self.remaining() < 1 {
            return Err(CodecError::UnexpectedEndOfInput);
        }
        Ok(self.data[self.read_pos])
    }

    /// Skip `n` bytes
    pub fn skip(&mut self, n: usize) -> CodecResult<()> {
        if self.remaining() < n {
            return Err(CodecError::UnexpectedEndOfInput);
        }
        self.read_pos += n;
        Ok(())
    }

    /// Reset read position to beginning
    pub fn reset(&mut self) {
        self.read_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_frame() {
        let data = [1, 2, 3, 4, 5];
        let frame = NeuralFrame::new(&data);

        assert_eq!(frame.len(), 5);
        assert!(!frame.is_empty());
        assert_eq!(frame.as_slice(), &data);
        assert_eq!(frame.channel_count_u16().unwrap(), 5);
    }

    #[test]
    fn test_neural_frame_empty() {
        let data: [i32; 0] = [];
        let frame = NeuralFrame::new(&data);

        assert_eq!(frame.len(), 0);
        assert!(frame.is_empty());
    }

    #[test]
    fn test_compressed_packet() {
        let mut buffer = [0u8; 100];
        let mut packet = CompressedPacket::new(&mut buffer);

        assert_eq!(packet.capacity(), 100);
        assert_eq!(packet.bytes_written(), 0);
        assert_eq!(packet.remaining(), 100);

        packet.advance(50).unwrap();
        assert_eq!(packet.bytes_written(), 50);
        assert_eq!(packet.remaining(), 50);
    }

    #[test]
    fn test_compressed_packet_overflow() {
        let mut buffer = [0u8; 10];
        let mut packet = CompressedPacket::new(&mut buffer);

        let result = packet.advance(20);
        assert!(matches!(result, Err(CodecError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_compressed_packet_split() {
        let mut buffer = [0u8; 100];
        // Write some test data
        buffer[0] = 1;
        buffer[1] = 2;

        let mut packet = CompressedPacket::new(&mut buffer);
        packet.advance(2).unwrap();

        let (written, remaining) = packet.split();
        assert_eq!(written, &[1, 2]);
        assert_eq!(remaining.len(), 98);
    }

    #[test]
    fn test_compressed_input() {
        let data = [1, 2, 3, 4, 5];
        let mut input = CompressedInput::new(&data);

        assert_eq!(input.len(), 5);
        assert_eq!(input.remaining(), 5);

        let byte = input.read_u8().unwrap();
        assert_eq!(byte, 1);
        assert_eq!(input.position(), 1);
        assert_eq!(input.remaining(), 4);
    }

    #[test]
    fn test_compressed_input_read_bytes() {
        let data = [1, 2, 3, 4, 5];
        let mut input = CompressedInput::new(&data);

        let bytes = input.read_bytes(3).unwrap();
        assert_eq!(bytes, &[1, 2, 3]);
        assert_eq!(input.remaining(), 2);
    }

    #[test]
    fn test_compressed_input_underflow() {
        let data = [1, 2, 3];
        let mut input = CompressedInput::new(&data);

        let result = input.read_bytes(5);
        assert_eq!(result, Err(CodecError::UnexpectedEndOfInput));
    }

    #[test]
    fn test_compressed_input_peek() {
        let data = [1, 2, 3];
        let mut input = CompressedInput::new(&data);

        let peeked = input.peek_u8().unwrap();
        assert_eq!(peeked, 1);
        assert_eq!(input.position(), 0); // Position unchanged

        let read = input.read_u8().unwrap();
        assert_eq!(read, 1);
        assert_eq!(input.position(), 1);
    }
}
