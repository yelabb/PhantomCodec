//! Demonstration of Fixed-Width Block Packing (PFOR) strategy
//!
//! Shows lossless compression with ultra-low latency characteristics.

use phantomcodec::{compress_fixed_width, decompress_fixed_width};

fn main() {
    println!("=== Fixed-Width Block Packing Demo ===\n");

    // Simulate realistic neural voltage data (small variations around baseline)
    let original = [
        2048, 2050, 2049, 2051, 2048, 2052, 2047, 2053, 2046, 2054, 2045, 2055, 2044, 2056, 2043,
        2057, 2042, 2058, 2041, 2059, 2040, 2060, 2039, 2061, 2038, 2062, 2037, 2063, 2036, 2064,
        2035, 2065,
    ];

    let mut compressed = [0u8; 200];
    let mut workspace = [0i32; 32];

    // Compress with fixed-width block packing
    let size = compress_fixed_width(&original, &mut compressed, &mut workspace)
        .expect("Compression failed");

    println!("Original size: {} bytes", original.len() * 4);
    println!("Compressed size: {} bytes", size);
    println!(
        "Compression ratio: {:.2}%",
        (size as f64 / (original.len() * 4) as f64) * 100.0
    );
    println!();

    // Decompress
    let mut decompressed = [0i32; 32];
    let count = decompress_fixed_width(&compressed[..size], &mut decompressed, &mut workspace)
        .expect("Decompression failed");

    assert_eq!(count, original.len());
    assert_eq!(&original[..], &decompressed[..]);

    println!("✓ Lossless roundtrip successful!");
    println!("✓ All {} samples match original values", count);
    println!();
    println!("Key Features:");
    println!("  • Lossless compression (unlike Packed4)");
    println!("  • <10µs decode latency (predictable, branchless)");
    println!("  • Variable bit-width per 32-sample block");
    println!("  • SIMD-friendly memory access patterns");
}
