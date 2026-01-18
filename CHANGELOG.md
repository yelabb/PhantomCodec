# Changelog

All notable changes to PhantomCodec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- <10μs decode latency on Cortex-M4F @ 168MHz
- 50% typical compression ratio on neural spike data
- Zero allocations in hot path (stack + static buffers only)
- Panic-free operation with strict compile-time safety

### Supported Platforms
- `no_std` embedded targets (ARM Cortex-M)
- Standard Rust with `std` feature
- Portable SIMD acceleration (when available)
- ARM DSP intrinsics support (Cortex-M4F)

[0.1.0]: https://github.com/yelabb/PhantomCodec/releases/tag/v0.1.0
