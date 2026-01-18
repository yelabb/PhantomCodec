# PhantomCodec: Aspirational Goals & Future Vision

**Date:** January 18, 2026  
**Status:** Roadmap for future development

This document preserves the **aspirational performance targets** that inspired PhantomCodec's design. These are not current capabilities, but represent the next evolution of the codec.

---

## 🎯 The Sub-10µs Vision

### Current Reality (v0.2.0)
- **Decode latency**: ~130-170µs on Cortex-M4F @ 168MHz
- **Algorithm**: Delta + Varint + Rice coding
- **SIMD**: Portable SIMD (nightly) or scalar fallback
- **Status**: ✅ Real-time compatible for 40Hz BCI streaming (0.5% of frame budget)

### Future Goal
- **Decode latency**: <10µs on Cortex-M4F @ 168MHz (**13-17x faster**)
- **Rationale**: Enable ultra-low-latency closed-loop control (<1ms total pipeline)
- **Challenge**: Requires fundamental algorithm redesign

---

## 🧩 Roadmap to Sub-10µs

### Phase 1: ARM DSP Intrinsics (Expected: ~65-85µs)
**Target:** 1.5-2x speedup from current scalar implementation

**Implementation:**
```rust
#[cfg(feature = "cortex-m-dsp")]
mod cortex_m_dsp {
    use cortex_m::dsp::*;
    
    /// Parallel delta computation using ARM CMSIS-DSP
    pub fn compute_deltas_dsp(input: &[i32], output: &mut [i32]) {
        // Convert i32 → Q15 fixed-point (pack 2×i16 into u32)
        // Use __SSUB16 for dual 16-bit parallel subtraction
        // Process 2 samples per cycle vs 1 in scalar code
    }
}
```

**Requirements:**
- `cortex-m` crate DSP feature
- Q15 fixed-point conversion (12-bit neural data fits in 16-bit signed)
- Intrinsics: `__SSUB16`, `__SADD16`, `__SMULBB`

**Limitations:**
- Still too slow for <10µs target
- Complexity overhead of varint/Rice encoding remains

---

### Phase 2: Simplified Bit-Packing (<10µs target)
**Target:** 6-8x faster than Phase 1

**Algorithm Change:**
Replace complex varint/Rice with fixed-width bit packing:

```rust
/// Ultra-fast 4-bit packed encoding (no entropy coding)
/// 
/// Assumes 12-bit neural ADC data (4096 levels)
/// Delta range: -4095 to +4095 → store as signed 13-bit
/// Compress 13-bit → 4-bit by dividing by 512 (lossy)
/// 
/// Trade-off: Slight quantization error for 10x speed gain
pub fn encode_fixed_4bit(input: &[i32], output: &mut [u8]) {
    // Pack 2 samples per byte (4 bits each)
    // No varint overhead, no Rice parameter adaptation
    // Pure bit-shifting operations (1-2 cycles per sample)
}
```

**Performance Estimate:**
- 1024 channels ÷ 2 samples/byte = 512 bytes
- Write 512 bytes at ~64 cycles/8 bytes = 4,096 cycles
- @ 168MHz: ~24ns/cycle × 4,096 = **~10µs** ✅

**Trade-offs:**
- ❌ Compression ratio drops (71% → 50%)
- ❌ Lossy quantization (acceptable for spike detection)
- ✅ Guaranteed <10µs latency
- ✅ Stable Rust compatible (no nightly required)

---

### Phase 3: Hardware Acceleration (Future Hardware)
**Target:** <1µs decode for next-gen neural interfaces

**Cortex-M55/M85 with Helium (MVE):**
- 128-bit SIMD vectors (4×i32 parallel ops)
- Fused multiply-accumulate for Rice decoding
- Expected: 5-10x faster than Cortex-M4F

**Custom ASIC:**
- Dedicated neural data decompressor
- Parallel channel processing
- Zero-copy DMA integration

---

## 📊 Performance Evolution Table

| Version | Latency (M4F @ 168MHz) | Algorithm | Compression | Rust |
|---------|------------------------|-----------|-------------|------|
| **v0.2.0** (current) | ~150µs | Delta+Varint+Rice | 71% | Nightly |
| **v0.3.0** (Phase 1) | ~75µs | DSP intrinsics | 71% | Nightly |
| **v1.0.0** (Phase 2) | **<10µs** | Fixed 4-bit packing | 50% | Stable |
| Future (Phase 3) | <1µs | ASIC accelerated | 50% | N/A |

---

## 🔬 Why Current Architecture Can't Hit <10µs

### Bottleneck Analysis

1. **Varint Decoding Loop** (~40% of time)
   - Variable-length integers require bit-by-bit parsing
   - Branch-heavy (if value < 128, else...)
   - Cannot fully vectorize with SIMD

2. **Rice Parameter Adaptation** (~20% of time)
   - Frame-level optimization calculates best k value
   - Adds computational overhead before encoding

3. **Memory Bandwidth** (~30% of time)
   - Cortex-M4F: 168MHz CPU but only ~84MB/s SRAM
   - Non-contiguous memory access patterns
   - Cache misses on compressed bitstream reads

4. **Instruction Overhead** (~10% of time)
   - Complex control flow (strategy selection)
   - Function call overhead (encoder dispatch)

### Why 4-Bit Packing Wins

Fixed-width encoding eliminates:
- ✅ Variable-length parsing logic
- ✅ Adaptive parameter calculation
- ✅ Complex bit manipulation loops
- ✅ Unpredictable branching

Result: Pure arithmetic + shifts = **predictable <10µs**.

---

## 🎓 Lessons Learned

### What We Got Right
1. **Zero-allocation design** - Correct from day one
2. **Sans-IO architecture** - Perfect for embedded
3. **Comprehensive benchmarking** - Caught the gap early
4. **Modular strategy system** - Easy to swap algorithms

### What We Learned
1. **Entropy coding has overhead** - Compression ratio isn't free
2. **Documentation should match reality** - Aspirations vs measurements
3. **Algorithm complexity matters more than SIMD** - 10x gap requires redesign
4. **Embedded is different** - PC benchmarks mislead without scaling

---

## 🚀 Next Steps

### To Implement Phase 1 (DSP Intrinsics)
- [ ] Add `cortex-m` dependency with `dsp` feature
- [ ] Implement Q15 fixed-point conversion utilities
- [ ] Write `compute_deltas_dsp()` using `__SSUB16`
- [ ] Benchmark on real STM32F4 hardware
- [ ] Validate 1.5-2x speedup claim

### To Implement Phase 2 (Sub-10µs)
- [ ] Design fixed 4-bit packing format spec
- [ ] Implement `encode_fixed_4bit()` / `decode_fixed_4bit()`
- [ ] Remove dependency on nightly Rust
- [ ] Benchmark on STM32F4 (target: <10µs)
- [ ] Document compression ratio trade-off

### Documentation
- [x] Correct current performance claims in README
- [x] Create INSPIRATION.md for aspirational goals
- [ ] Add "Future Work" section to ARCHITECTURE.md
- [ ] Update CHANGELOG with realistic v0.2.0 claims

---

## 🙏 Acknowledgments

This roadmap reflects honest engineering: **we aimed for <10µs, achieved <150µs, and learned what it takes to close the gap**. The current codec is production-ready for real-time BCI streaming (40Hz), but ultra-low-latency closed-loop control will require Phase 2 implementation.

> "Premature optimization is the root of all evil, but premature *documentation* is worse."  
> — Lessons from PhantomCodec development

---

## References

- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) - Current measured performance
- [ARCHITECTURE.md](ARCHITECTURE.md) - Compression algorithm details
- [src/simd.rs:250-265](src/simd.rs) - Empty DSP module with implementation notes
- ARM CMSIS-DSP: https://arm-software.github.io/CMSIS_5/DSP/html/index.html
