# Changelog

All notable changes to PhantomCodec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **[NEW]** ARM Helium (MVE) support for Cortex-M55/M85 processors
  - Added `mve` feature flag for 128-bit SIMD operations
  - Implemented `compute_deltas_mve()` for 8x i16 parallel delta computation
  - Implemented `reconstruct_from_deltas_mve()` with cascading vector operations
  - Implemented `zigzag_encode_mve()` and `zigzag_decode_mve()` for 8x16-bit ZigZag encoding
  - Implemented `unpack4_mve()` for ultra-fast 4-bit unpacking (32 samples per iteration)
  - Implemented `pack4_mve()` for 4-bit packing with Helium
  - Implemented `unpack_fixed_width_mve()` for bit-parallel unpacking (4-16 bit widths)
  - Added `.cargo/config.toml` with build configuration for `thumbv8.1m.main-none-eabihf` target
  - Added 21 comprehensive tests for MVE functions
  - Target performance: <3µs decode latency for 1024 channels (10x faster than scalar)
  - Functions use scalar fallbacks until `core::arch::arm` MVE intrinsics stabilize
  - See README for usage instructions and performance targets

### Changed - BREAKING
- **[CRITICAL SAFETY FIX]** All compression and decompression functions now require a `workspace: &mut [i32]` parameter
  - `compress_spike_counts()`: Added `workspace` parameter
  - `decompress_spike_counts()`: Added `workspace` parameter  
  - `compress_voltage()`: Added `workspace` parameter
  - `decompress_voltage()`: Added `workspace` parameter

### Fixed
- **[CRITICAL]** Eliminated unsafe static mutable buffer that caused data corruption in interrupt contexts
  - Previous implementation used `static mut TEMP_BUFFER` which could be accessed by multiple execution contexts simultaneously
  - This caused immediate data corruption if compression was called from both main loop and interrupt handlers (e.g., ADC DMA completion)
  - Now requires caller to provide workspace buffer, ensuring each context uses separate memory
  - See README "Safety & Reentrancy" section for detailed explanation and best practices

- **[CRITICAL]** Fixed `BitWriter` dirty buffer corruption bug
  - `BitWriter::write_bits()` previously only set bits to 1 using `|=` but never cleared bits to 0
  - When reusing buffers (common in embedded to save RAM), old data would corrupt new compressed streams
  - Decompression would fail or produce garbage data
  - Now properly clears each byte before writing first bit to it, ensuring clean writes regardless of buffer state
  - Added tests: `test_dirty_buffer_reuse()` and `test_partial_byte_dirty_buffer()` to prevent regression

- **[CRITICAL]** Fixed silent data loss in Rice encoding
  - `rice_encode()` previously clamped quotient values to 255, silently truncating large deltas
  - This converted a "lossless" codec into a lossy one without warning
  - Large neural artifacts or movement noise would be corrupted during compression
  - Decompression would produce vastly smaller values than the original, causing downstream analysis failures
  - Now returns `RiceQuotientOverflow` error when values exceed safe encoding limits
  - Added tests: `test_rice_quotient_overflow_detection()` and `test_rice_array_with_overflow()`
  - Added constant `MAX_RICE_QUOTIENT = 255` for clarity

### Migration Guide
```rust
// Old API (unsafe):
let size = compress_spike_counts(&data, &mut output)?;

// New API (safe):
let mut workspace = [0i32; 1024];  // Allocate once per execution context
let size = compress_spike_counts(&data, &mut output, &mut workspace)?;
```

## [0.1.0] - 2026-01-18

### Added
- Initial release of PhantomCodec
- Core compression trait architecture with `CompressionStrategy`
- Delta + Varint encoding for unsigned spike counts
- Adaptive Rice coding for signed voltage data
- Zero-copy buffer abstractions (`NeuralFrame`, `CompressedPacket`)
- Bit-level writer/reader with varint and zigzag support
- SIMD-accelerated delta computation (portable SIMD + scalar fallback)
- Compile-time ZigZag elimination via trait constants
- Adaptive Rice parameter selection using MAD heuristic
- Comprehensive error handling with `CodecError` enum
- High-level API: `compress_spike_counts()`, `decompress_spike_counts()`
- `#![no_std]` compatibility for embedded targets
- STM32 DMA integration example
- Full test coverage for all modules
- Documentation with examples and performance characteristics

### Performance
- <150μs decode latency on Cortex-M4F @ 168MHz (measured: ~130-170μs)
- 71% compression ratio on neural spike data (exceeds 50% target)
- Zero allocations in hot path (stack + static buffers only)
- Panic-free operation with strict compile-time safety
- Future goal: Sub-10μs latency with ARM DSP intrinsics (see INSPIRATION.md)

### Supported Platforms
- `no_std` embedded targets (ARM Cortex-M)
- Standard Rust with `std` feature
- Portable SIMD acceleration (when available)
- ARM DSP intrinsics support (Cortex-M4F)

[0.1.0]: https://github.com/yelabb/PhantomCodec/releases/tag/v0.1.0
