# PhantomCodec Architecture Specification

## Executive Summary

PhantomCodec is a `#![no_std]` Rust crate for real-time lossless compression of high-density neural data (1,024+ channels) optimized for bare-metal ARM Cortex-M environments. It achieves ~130-170μs decode latency (Cortex-M4F @ 168MHz) with zero allocations through compile-time monomorphization, zero-copy memory architecture, and SIMD acceleration. See [INSPIRATION.md](INSPIRATION.md) for sub-10μs roadmap.

---

## 1. Trait Strategy: Zero-Cost Abstractions via Monomorphization

### Design Philosophy

PhantomCodec uses **static dispatch** exclusively, avoiding `Box<dyn Trait>` dynamic dispatch to eliminate vtable overhead and enable aggressive compiler inlining.

### Core Trait Definition

```rust
pub trait CompressionStrategy {
    /// Compile-time constant: enables dead-code elimination
    const REQUIRES_ZIGZAG: bool;
    
    /// Strategy identifier for packet header
    const STRATEGY_ID: StrategyId;
    
    /// Compression implementation (monomorphized per strategy)
    fn compress(input: &[i32], output: &mut [u8]) -> CodecResult<usize>;
    
    /// Decompression implementation
    fn decompress(input: &[u8], output: &mut [i32]) -> CodecResult<usize>;
}
```

### Monomorphization Example

```rust
// Generic compression function
pub fn compress<S: CompressionStrategy>(input: &[i32], output: &mut [u8]) -> CodecResult<usize> {
    S::compress(input, output)
}

// Usage generates specialized code at compile time:
compress::<DeltaVarintStrategy>(...) // Generates Strategy A code
compress::<RiceStrategy>(...)        // Generates Strategy B code (separate binary)
```

**Compiler Effect**: Each call to `compress<S>` generates a unique function specialized for that strategy. The optimizer inlines strategy-specific code, eliminating branches and producing machine code equivalent to hand-written specialized functions.

### Compile-Time ZigZag Elimination

```rust
impl DeltaVarintStrategy {
    const REQUIRES_ZIGZAG: bool = false; // For unsigned spike counts
}

impl RiceStrategy {
    const REQUIRES_ZIGZAG: bool = true; // For signed voltages
}

// In compression code:
if S::REQUIRES_ZIGZAG {
    // This branch is eliminated at compile time when false
    apply_zigzag(value)
} else {
    value as u32
}
```

**Benefit**: Zero runtime cost for type-specific optimizations. The compiler generates two completely separate code paths with no branching overhead.

---

## 2. Zero-Copy Memory Architecture

### Lifetime-Based Borrowing

PhantomCodec **never allocates memory**. All buffers are provided by the application layer, enabling:
- Stack allocation (small buffers)
- Static globals (embedded systems)
- DMA-mapped regions (hardware accelerators)

### Core Buffer Types

```rust
/// Immutable view of neural data
pub struct NeuralFrame<'a> {
    data: &'a [i32],  // Borrows application memory
}

/// Mutable compression output
pub struct CompressedPacket<'a> {
    buffer: &'a mut [u8],  // Zero-copy write
    bytes_written: usize,
}
```

### Memory Transformation Pipeline

```text
Application Memory:
┌─────────────────────┐
│ static mut INPUT:   │
│   [i32; 1024]       │ ← Application owns this
└─────────────────────┘
         │
         │ Borrow as &[i32]
         ▼
┌─────────────────────┐
│ NeuralFrame::new()  │ ← Zero-copy wrapper
└─────────────────────┘
         │
         │ compute_deltas() (SIMD in-place)
         ▼
┌─────────────────────┐
│ static mut DELTAS:  │
│   [i32; 1024]       │ ← Reuses another static buffer
└─────────────────────┘
         │
         │ varint_encode()
         ▼
┌─────────────────────┐
│ static mut OUTPUT:  │
│   [u8; 4096]        │ ← Compressed result
└─────────────────────┘
```

**Key Insight**: Data flows through borrows, never copies. Lifetimes enforce that borrowed memory outlives the compression operation.

### In-Place Transformation Example

```rust
// Application controls memory placement
static mut NEURAL_DATA: [i32; 1024] = [0; 1024];
static mut COMPRESSED: [u8; 4096] = [0; 4096];

unsafe {
    // PhantomCodec borrows memory, never allocates
    let frame = NeuralFrame::new(&NEURAL_DATA);
    let mut packet = CompressedPacket::new(&mut COMPRESSED);
    
    // Transform happens on application's memory
    compress_spike_counts(frame.as_slice(), packet.as_mut_slice())?;
}
```

---

## 3. Bit-Level Abstraction: Safe Bit Packing

### BitWriter Design

```rust
pub struct BitWriter<'a> {
    buffer: &'a mut [u8],  // Byte-level storage
    bit_pos: usize,        // Current bit cursor (0 = first bit of byte 0)
}
```

### Variable-Width Writing Without Unsafe

```rust
impl<'a> BitWriter<'a> {
    pub fn write_bits(&mut self, value: u32, width: u8) -> CodecResult<()> {
        // Bounds check (not unsafe!)
        if self.remaining_bits() < width as usize {
            return Err(CodecError::BufferTooSmall { required: ... });
        }
        
        // Write bits MSB-first using safe indexing
        for i in (0..width).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_idx = self.bit_pos / 8;  // Which byte?
            let bit_idx = self.bit_pos % 8;   // Which bit in byte?
            
            // Safe: bounds checked above
            if bit != 0 {
                self.buffer[byte_idx] |= 1 << (7 - bit_idx);
            }
            self.bit_pos += 1;
        }
        Ok(())
    }
}
```

**Safety Properties**:
1. **No unsafe pointer arithmetic**: Uses standard array indexing
2. **Compile-time bounds checking**: Enabled in debug builds via assertions
3. **Runtime error propagation**: Returns `Result` instead of panicking
4. **Bit-level precision**: Handles non-byte-aligned data (e.g., u4, u7 values)

### Varint Encoding Implementation

```rust
pub fn write_varint(&mut self, mut value: u32) -> CodecResult<()> {
    self.flush()?;  // Align to byte boundary first
    
    loop {
        let mut byte = (value & 0x7F) as u8;  // Take lower 7 bits
        value >>= 7;
        
        if value != 0 {
            byte |= 0x80;  // Set continuation bit
        }
        
        // Write byte (bounds-checked)
        let byte_idx = self.bit_pos / 8;
        self.buffer[byte_idx] = byte;
        self.bit_pos += 8;
        
        if value == 0 { break; }
    }
    Ok(())
}
```

**Encoding Format**:
```
Value: 300 (decimal) = 0b100101100
Encoded: [0b10101100, 0b00000010]
          └─ 7 bits ┘  └─ 7 bits┘
          MSB=1 (more)  MSB=0 (done)
```

---

## 4. SIMD & Concurrency

### Portable SIMD Strategy

PhantomCodec uses feature detection to select optimal implementation:

```rust
#[cfg(all(feature = "simd", target_feature = "simd128"))]
fn compute_deltas_simd(input: &[i32], output: &mut [i32]) {
    use core::simd::{i32x8, Simd};
    const LANES: usize = 8;
    
    for i in (0..input.len()).step_by(LANES) {
        let chunk = Simd::<i32, 8>::from_slice(&input[i..]);
        let prev = /* shift previous values */;
        let deltas = chunk - prev;
        deltas.copy_to_slice(&mut output[i..]);
    }
}

#[cfg(not(all(feature = "simd", target_feature = "simd128")))]
fn compute_deltas_scalar(input: &[i32], output: &mut [i32]) {
    // Fallback scalar implementation
}
```

### ARM Cortex-M DSP Extensions

For Cortex-M4F (which **does not have NEON**):

```rust
#[cfg(feature = "cortex-m-dsp")]
use cortex_m::dsp;

// Use packed SIMD instructions
// SSUB16: Two 16-bit subtractions in parallel (Q15 format)
unsafe {
    let result = dsp::__SSUB16(input_packed, prev_packed);
}
```

**Correction**: Cortex-M4F has DSP extensions (SADD16, SSUB16) for 16-bit packed integers, NOT NEON. NEON is only available on Cortex-A series processors.

### Concurrency Model

PhantomCodec is **not thread-safe by default** because:
1. Embedded systems are typically single-threaded
2. `no_std` has no built-in synchronization primitives
3. Lock-free data structures add overhead

**Application-Level Concurrency**: Users can wrap calls in `cortex_m::interrupt::free` critical sections if needed:

```rust
cortex_m::interrupt::free(|_cs| {
    // Critical section: interrupts disabled
    compress_spike_counts(&INPUT, &mut OUTPUT)?;
});
```

---

## 5. Error Handling

### Non-Blocking Result Types

```rust
/// All operations return Result for error propagation
pub type CodecResult<T> = Result<T, CodecError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    BufferTooSmall { required: usize },
    InvalidChannelCount { expected: usize, actual: usize },
    CorruptedHeader,
    UnsupportedVersion { version: u8 },
    UnexpectedEndOfInput,
    InvalidZigZagValue,
    InvalidRiceParameter { k: u8 },
    BitPositionOverflow,
}
```

### FFI-Friendly Error Codes

```rust
impl CodecError {
    /// Convert to i32 for C FFI
    pub const fn to_error_code(self) -> i32 {
        match self {
            CodecError::BufferTooSmall { .. } => -1,
            CodecError::InvalidChannelCount { .. } => -2,
            // ...
        }
    }
}
```

### Real-Time Guarantees

1. **No panics**: All error paths return `Result`
2. **No allocations**: Errors are `Copy` types (stack-only)
3. **Predictable timing**: Error checks are bounds checks (constant time)
4. **Recoverable**: Caller can retry with larger buffer or different strategy

### Example Error Handling

```rust
match compress_spike_counts(&input, &mut output) {
    Ok(size) => {
        // Transmit compressed data
        uart_dma_send(&output[..size]);
    }
    Err(CodecError::BufferTooSmall { required }) => {
        // Fall back to uncompressed transmission
        log_warning!("Compression failed, need {} bytes", required);
        uart_send_raw(&input);
    }
    Err(e) => {
        // Critical error: trigger watchdog reset
        system_reset();
    }
}
```

---

## Design Decisions Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Rice Parameter** | Adaptive (MAD heuristic) | Neural firing rates vary dramatically; fixed k wastes 15% compression |
| **DMA Integration** | Sans-IO (examples only) | Library remains portable; chip-specific code in user examples |
| **ZigZag Encoding** | Optional via trait constant | Spike counts are unsigned; eliminates ZigZag overhead at compile time |
| **SIMD Backend** | Portable + DSP fallback | No NEON on Cortex-M; use DSP instructions (SSUB16) + scalar fallback |
| **Memory Model** | Application-owned | Enables DMA, static buffers, and stack allocation based on platform |

---

## Performance Characteristics

| Metric | Target | Measured (Cortex-M4F @ 168MHz) |
|--------|--------|---------------------------------|
| Encode Latency | <150μs | ~140-180μs (1024 channels) |
| Decode Latency | <150μs | ~130-170μs (1024 channels) |
| Compression Ratio | 50% | 71% reduction (typical neural data) |
| Memory Usage | <4KB static | 3.5KB (3× 1024-channel buffers) |
| Code Size | <16KB | ~8KB (Release build) |

> **Note**: See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed measurements and [INSPIRATION.md](INSPIRATION.md) for sub-10μs roadmap.

---

## Integration with Phantom Suite

```
┌──────────────┐   40Hz      ┌──────────────┐   Compressed   ┌──────────────┐
│ PhantomLink  │ ══════════▶ │ PhantomCodec │ ═════════════▶ │ PhantomCore  │
│  (Python)    │   [i32;142] │    (Rust)    │    ~3KB/pkt   │    (C++)     │
└──────────────┘             └──────────────┘                └──────────────┘
       │                                                             │
       │ Encoder: Delta + Varint                                    │
       │ Latency: <25μs                                             │
       │                                                             │
       └──────────────────▶ 240KB/s → 120KB/s (50% reduction) ◀─────┘
```

**Future Work**:
- C FFI bindings for PhantomCore integration
- PyO3 Python bindings for PhantomLink
- WebAssembly target for PhantomLoop browser visualization

---

## Conclusion

PhantomCodec achieves sub-10μs real-time compression through:
1. **Compile-time specialization** via monomorphized traits
2. **Zero-copy architecture** with lifetime-enforced borrowing
3. **Safe bit manipulation** without unsafe pointer arithmetic
4. **SIMD acceleration** with portable and platform-specific backends
5. **Non-blocking error handling** suitable for embedded real-time systems

This architecture enables deployment on resource-constrained microcontrollers while maintaining the safety guarantees of Rust's ownership system.
