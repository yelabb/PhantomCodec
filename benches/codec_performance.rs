use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
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

/// Benchmark decompression - THIS IS THE CRITICAL <10μs TARGET
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
        
        let compressed_size = compress_spike_counts(
            &spike_counts,
            &mut compressed,
            &mut workspace,
        )
        .unwrap();
        
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
    let mut group = c.benchmark_group("compression_ratio");
    
    for &num_channels in &[128, 256, 512, 1024] {
        let mut spike_counts = vec![0i32; num_channels];
        
        // Simulate realistic sparse neural data
        for i in (0..num_channels).step_by(20) {
            spike_counts[i] = (i % 15) as i32;
        }
        
        let mut compressed = vec![0u8; num_channels * 8];
        let mut workspace = vec![0i32; num_channels];
        
        let compressed_size = compress_spike_counts(
            &spike_counts,
            &mut compressed,
            &mut workspace,
        )
        .unwrap();
        
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
            self.state = oldstate.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
            let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
            let rot = (oldstate >> 59) as u32;
            xorshifted.rotate_right(rot)
        }
    }
    
    // Generate high-entropy random spike counts (0-15 range)
    let mut rng = Pcg32::new(42);
    let mut spike_counts = vec![0i32; num_channels];
    for i in 0..num_channels {
        spike_counts[i] = (rng.next() % 16) as i32; // Random 0-15
    }
    
    let mut compressed = vec![0u8; num_channels * 8];
    let mut workspace = vec![0i32; num_channels];
    let mut decompressed = vec![0i32; num_channels];
    
    // Compress once to get size
    let compressed_size = compress_spike_counts(
        &spike_counts,
        &mut compressed,
        &mut workspace,
    )
    .unwrap();
    
    println!("Worst-case (random data) compression: {:.1}%", 
        (compressed_size as f64 / (num_channels * 4) as f64) * 100.0);
    
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

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_compression_ratio,
    bench_dense_data
);
criterion_main!(benches);
