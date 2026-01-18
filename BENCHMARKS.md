# PhantomCodec Benchmarks

This document explains the benchmark results for PhantomCodec.

## Running Benchmarks

```bash
# Standard benchmarks (release mode)
cargo bench

# With HTML reports
cargo bench --features std

# View results
open target/criterion/report/index.html  # macOS
start target/criterion/report/index.html # Windows
xdg-open target/criterion/report/index.html # Linux
```

## About Performance Claims

### Realistic Embedded Targets

**Cortex-M4F @ 168MHz expectations:**
- 1024 channels: **<150µs decode latency**
- Algorithm is O(n) with varint decoding loops
- For <10µs requirement: Need simpler algorithm (e.g., 4-bit packing)

### What Benchmarks Show

1. **Trends**: Linear scaling, decompression faster than compression
2. **Ratios**: Compression efficiency (hardware-independent)
3. **Relative Performance**: Algorithm characteristics

### On-Target Profiling Required

Desktop benchmarks provide algorithmic insight but **cannot predict embedded timing**.
Real measurements require:
- STM32F4 development board
- Hardware timers (DWT cycle counter)
- Realistic memory configuration (SRAM, cache)

## What to Look For

### ✅ Good Signs
- Decompression faster than compression
- <50% compression ratio on sparse data
- Linear scaling with channel count
- Consistent timing across runs

### ⚠️ Warning Signs
- Decompression slower than compression
- >80% compression ratio (barely compressing)
- Non-linear scaling
- High variance in timing

## Benchmark Suite

### 1. Compression Performance
Tests compression time for 128, 256, 512, 1024 channels

### 2. Decompression Performance  
**CRITICAL**: Tracks progress toward real-time embedded performance (<150µs measured, <10µs future goal)

### 3. Compression Ratio
Shows actual compression percentage

### 4. Dense Data Test
Worst-case scenario: all channels firing

## Interpreting Results

Desktop benchmarks show:
```
decompression/1024: 2.57 μs
```

**This does NOT mean 2.57µs on embedded hardware.**

Embedded performance depends on:
- Memory latency (SRAM vs DRAM)
- Cache configuration
- Instruction timing (ARM vs x86)
- Compiler optimization

**Realistic embedded estimate:** 150-200µs for 1024 channels on M4F.

The important metrics are **trends** and **compression ratios**.
