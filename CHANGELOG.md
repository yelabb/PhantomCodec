# Changelog

All notable changes to PhantomCodec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Linear Predictive Coding (LPC)** for improved compression on smooth signals
  - `compress_voltage_lpc2()`: 2nd-order predictor (models constant velocity)
    - Expected compression: ~55-60% (vs 71% with Delta)
    - Best for: Wavy LFP signals, smooth neural data
    - Residuals: ~±50 (vs ~±200 with Delta)
  - `compress_voltage_lpc3()`: 3rd-order predictor (models constant acceleration)
    - Expected compression: ~50-55% (vs 71% with Delta)
    - Best for: Signals with smooth curvature
    - Residuals: ~±30 (vs ~±200 with Delta)
  - `PredictorMode` enum: `Delta`, `LPC2`, `LPC3`, `Reserved`
  - Automatic predictor detection in `decompress_voltage()`
  - See `src/lpc.rs` for mathematical background
  
- **Enhanced PacketHeader** with predictor mode support
  - Byte 7 layout: `[Strategy:4|Predictor:2|Rice k:2]`
  - Backward compatible with existing Delta encoding
  - 2 bits for predictor mode (4 possible modes)

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
