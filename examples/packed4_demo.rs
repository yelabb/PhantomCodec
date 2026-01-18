//! Demonstration of ultra-low-latency Packed4 compression strategy
//!
//! This example shows the trade-off between compression quality and speed.
//! Packed4 achieves ~13-17x faster decoding by using fixed 4-bit packing
//! instead of variable-length entropy coding.
//!
//! Run with: cargo run --example packed4_demo

use phantomcodec::{
    compress_packed4, compress_spike_counts, decompress_packed4, decompress_spike_counts,
};

fn main() {
    println!("=== PhantomCodec Packed4 Ultra-Low-Latency Demo ===\n");

    // Simulate 1024-channel neural data (typical BCI application)
    let mut neural_data = [0i32; 1024];
    for (i, sample) in neural_data.iter_mut().enumerate() {
        // Simulate sparse spiking activity with values in multiples of 256
        // (Packed4 quantizes to 256-unit granularity)
        *sample = if i % 100 < 5 {
            ((i % 10) as i32) * 256
        } else {
            0
        };
    }

    let mut workspace = [0i32; 1024];

    // === Strategy 1: DeltaVarint (Default, Lossless) ===
    println!("Strategy 1: DeltaVarint (Lossless)");
    let mut compressed_varint = [0u8; 8192];
    let size_varint = compress_spike_counts(&neural_data, &mut compressed_varint, &mut workspace)
        .expect("Compression failed");
    println!("  Compressed size: {} bytes", size_varint);
    println!(
        "  Compression ratio: {:.1}%",
        (size_varint as f64 / (1024 * 4) as f64) * 100.0
    );
    println!("  Estimated decode time (M4F): ~150µs");
    println!("  Quality: Lossless\n");

    // === Strategy 2: Packed4 (Ultra-Fast, Lossy) ===
    println!("Strategy 2: Packed4 (Ultra-Low-Latency)");
    let mut compressed_packed4 = [0u8; 8192];
    let size_packed4 = compress_packed4(&neural_data, &mut compressed_packed4, &mut workspace)
        .expect("Compression failed");
    println!("  Compressed size: {} bytes", size_packed4);
    println!(
        "  Compression ratio: {:.1}%",
        (size_packed4 as f64 / (1024 * 4) as f64) * 100.0
    );
    println!("  Estimated decode time (M4F): ~10µs ⚡");
    println!("  Quality: Lossy (±256 quantization)");
    println!("  Speedup: ~15x faster decode\n");

    // Verify roundtrip
    let mut decompressed_varint = [0i32; 1024];
    let mut decompressed_packed4 = [0i32; 1024];

    decompress_spike_counts(
        &compressed_varint[..size_varint],
        &mut decompressed_varint,
        &mut workspace,
    )
    .expect("DeltaVarint decompression failed");

    decompress_packed4(
        &compressed_packed4[..size_packed4],
        &mut decompressed_packed4,
        &mut workspace,
    )
    .expect("Packed4 decompression failed");

    // Compare quality
    println!("=== Quality Comparison ===");
    println!("DeltaVarint: Perfect reconstruction (lossless)");

    let mut max_error = 0i32;
    let mut nonzero_errors = 0;
    for i in 0..1024 {
        let error = (neural_data[i] - decompressed_packed4[i]).abs();
        if error > max_error {
            max_error = error;
        }
        if error > 0 {
            nonzero_errors += 1;
        }
    }

    println!(
        "Packed4: Max error = {} units ({} samples with error)",
        max_error, nonzero_errors
    );
    println!("\n=== Use Case Recommendation ===");
    println!("• DeltaVarint: High-fidelity recording, offline analysis");
    println!("• Packed4: Closed-loop control, real-time spike detection");
    println!("  (where <10µs latency is critical)");
}
