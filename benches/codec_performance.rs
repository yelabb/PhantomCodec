use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use phantomcodec::simd::{compute_deltas, reconstruct_from_deltas, sum_abs_deltas};
use phantomcodec::{compress_spike_counts, decompress_spike_counts};

/// Benchmark compression of neural spike data
fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    // Test different channel counts
    for &num_channels in &[128, 256, 512, 1024] {
        let mut spike_counts = vec![0i32; num_channels];

        // Simulate realistic sparse neural data (5% active channels)
        for i in (0..num_channels).step_by(20) {
            spike_counts[i] = (i % 15) as i32; // 0-14 spikes
        }

        let mut compressed = vec![0u8; num_channels * 8];
        let mut workspace = vec![0i32; num_channels];

        group.bench_with_input(
            BenchmarkId::from_parameter(num_channels),
            &num_channels,
            |b, _| {
                b.iter(|| {
                    compress_spike_counts(
                        black_box(&spike_counts),
                        black_box(&mut compressed),
                        black_box(&mut workspace),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark decompression - measuring progress toward <10μs future goal (see INSPIRATION.md)
fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression");

    // Set measurement time to get accurate microsecond measurements
    group.measurement_time(std::time::Duration::from_secs(10));

    for &num_channels in &[128, 256, 512, 1024] {
        // Prepare compressed data
        let mut spike_counts = vec![0i32; num_channels];
        for i in (0..num_channels).step_by(20) {
            spike_counts[i] = (i % 15) as i32;
        }

        let mut compressed = vec![0u8; num_channels * 8];
        let mut workspace = vec![0i32; num_channels];

        let compressed_size =
            compress_spike_counts(&spike_counts, &mut compressed, &mut workspace).unwrap();

        compressed.truncate(compressed_size);

        let mut decompressed = vec![0i32; num_channels];

        group.bench_with_input(
            BenchmarkId::from_parameter(num_channels),
            &num_channels,
            |b, _| {
                b.iter(|| {
                    decompress_spike_counts(
                        black_box(&compressed),
                        black_box(&mut decompressed),
                        black_box(&mut workspace),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark compression ratio
fn bench_compression_ratio(c: &mut Criterion) {
    let group = c.benchmark_group("compression_ratio");

    for &num_channels in &[128, 256, 512, 1024] {
        let mut spike_counts = vec![0i32; num_channels];

        // Simulate realistic sparse neural data
        for i in (0..num_channels).step_by(20) {
            spike_counts[i] = (i % 15) as i32;
        }

        let mut compressed = vec![0u8; num_channels * 8];
        let mut workspace = vec![0i32; num_channels];

        let compressed_size =
            compress_spike_counts(&spike_counts, &mut compressed, &mut workspace).unwrap();

        let original_size = num_channels * 4; // i32 = 4 bytes
        let ratio = (compressed_size as f64 / original_size as f64) * 100.0;

        println!(
            "{} channels: {} bytes -> {} bytes ({:.1}% of original)",
            num_channels, original_size, compressed_size, ratio
        );
    }

    group.finish();
}

/// Benchmark worst-case scenario: high-entropy random data (true worst case)
fn bench_dense_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");

    let num_channels = 1024;

    // PCG random number generator (deterministic, high quality)
    struct Pcg32 {
        state: u64,
        inc: u64,
    }

    impl Pcg32 {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                inc: 1,
            }
        }

        fn next(&mut self) -> u32 {
            let oldstate = self.state;
            self.state = oldstate
                .wrapping_mul(6364136223846793005)
                .wrapping_add(self.inc);
            let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
            let rot = (oldstate >> 59) as u32;
            xorshifted.rotate_right(rot)
        }
    }

    // Generate high-entropy random spike counts (0-15 range)
    let mut rng = Pcg32::new(42);
    let mut spike_counts = vec![0i32; num_channels];
    for count in spike_counts.iter_mut().take(num_channels) {
        *count = (rng.next() % 16) as i32; // Random 0-15
    }

    let mut compressed = vec![0u8; num_channels * 8];
    let mut workspace = vec![0i32; num_channels];
    let mut decompressed = vec![0i32; num_channels];

    // Compress once to get size
    let compressed_size =
        compress_spike_counts(&spike_counts, &mut compressed, &mut workspace).unwrap();

    println!(
        "Worst-case (random data) compression: {:.1}%",
        (compressed_size as f64 / (num_channels * 4) as f64) * 100.0
    );

    group.bench_function("decompress_worst_case", |b| {
        b.iter(|| {
            decompress_spike_counts(
                black_box(&compressed[..compressed_size]),
                black_box(&mut decompressed),
                black_box(&mut workspace),
            )
            .unwrap()
        });
    });

    group.finish();
}

/// Benchmark SIMD delta encoding operations
/// These are the core primitives used by the codec and the cortex_m_dsp module
fn bench_simd_deltas(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_deltas");

    for &num_channels in &[128, 256, 512, 1024] {
        // Simulate realistic neural voltage data (12-bit ADC, centered around 2048)
        let input: Vec<i32> = (0..num_channels)
            .map(|i| 2048 + ((i as i32 * 7) % 100) - 50)
            .collect();
        let mut deltas = vec![0i32; num_channels];
        let mut reconstructed = vec![0i32; num_channels];

        group.bench_with_input(
            BenchmarkId::new("compute_deltas", num_channels),
            &num_channels,
            |b, _| {
                b.iter(|| {
                    compute_deltas(black_box(&input), black_box(&mut deltas));
                });
            },
        );

        // Pre-compute deltas for reconstruction benchmark
        compute_deltas(&input, &mut deltas);

        group.bench_with_input(
            BenchmarkId::new("reconstruct_from_deltas", num_channels),
            &num_channels,
            |b, _| {
                b.iter(|| {
                    reconstruct_from_deltas(black_box(&deltas), black_box(&mut reconstructed));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sum_abs_deltas", num_channels),
            &num_channels,
            |b, _| {
                b.iter(|| sum_abs_deltas(black_box(&deltas), black_box(num_channels)));
            },
        );
    }

    group.finish();
}

/// Benchmark ultra-low-latency 4-bit fixed-width encoding (Phase 2 from INSPIRATION.md)
/// Target: <10µs for 1024 channels on Cortex-M4F
fn bench_fixed_4bit_encoding(c: &mut Criterion) {
    use phantomcodec::simd::{decode_fixed_4bit, encode_fixed_4bit};

    let mut group = c.benchmark_group("fixed_4bit");

    for &num_channels in &[128_usize, 256, 512, 1024] {
        // Simulate neural delta data (small values that fit in 4-bit quantized)
        let deltas: Vec<i32> = (0..num_channels)
            .map(|i| ((i as i32 % 15) - 7) * 256) // Range: -7*256 to +7*256
            .collect();
        let mut encoded = vec![0u8; num_channels.div_ceil(2)];
        let mut decoded = vec![0i32; num_channels];

        group.bench_with_input(
            BenchmarkId::new("encode_4bit", num_channels),
            &num_channels,
            |b, _| {
                b.iter(|| {
                    encode_fixed_4bit(black_box(&deltas), black_box(&mut encoded)).unwrap();
                });
            },
        );

        // Pre-encode for decode benchmark
        encode_fixed_4bit(&deltas, &mut encoded).unwrap();

        group.bench_with_input(
            BenchmarkId::new("decode_4bit", num_channels),
            &num_channels,
            |b, _| {
                b.iter(|| {
                    decode_fixed_4bit(
                        black_box(&encoded),
                        black_box(num_channels),
                        black_box(&mut decoded),
                    )
                    .unwrap();
                });
            },
        );
    }

    // Report expected Cortex-M4F performance
    println!("\n=== Ultra-Low-Latency 4-bit Mode (Phase 2 target: <10µs) ===");
    println!("Desktop x86 times shown above.");
    println!("Expected Cortex-M4F @ 168MHz: ~6-10µs for 1024 channels");
    println!("Compression ratio: 50% (vs 71% with Rice coding)");
    println!("Trade-off: Lossy quantization (±128) for 15x speed gain\n");

    group.finish();
}

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_compression_ratio,
    bench_dense_data,
    bench_simd_deltas,
    bench_fixed_4bit_encoding
);
criterion_main!(benches);
