//! Demonstration of QoS auto-throttling for adaptive bandwidth management
//!
//! This example shows how the QoS system automatically adjusts compression
//! quality to maintain target bandwidth while preserving data fidelity.
//!
//! Run with: cargo run --example qos_demo

use phantomcodec::qos::{QosConfig, QosController, QualityLevel};
use phantomcodec::{compress_adaptive, decompress_adaptive};

fn main() {
    println!("=== PhantomCodec QoS Auto-Throttling Demo ===\n");

    // Create QoS configuration for a wireless BCI implant
    let config = QosConfig::new(100_000); // Target 100 KB/s
    let mut controller = QosController::new(config);

    println!("Configuration:");
    println!("  Target bandwidth: {} bytes/s", config.target_bandwidth);
    println!("  Max buffer: {} bytes", config.max_buffer_bytes);
    println!("  Priority: {:?}", config.priority);
    println!("  Min quality: {:?}\n", config.min_quality);

    // Simulate different neural activity levels
    let scenarios = [
        ("Low activity (resting)", 50, 1024),
        ("Medium activity (movement)", 150, 1024),
        ("High activity (seizure)", 350, 1024),
    ];

    let mut workspace = [0i32; 1024];

    for (name, spike_density, num_channels) in scenarios {
        println!("=== Scenario: {} ===", name);

        // Generate synthetic neural data
        let mut neural_data = [0i32; 1024];
        for (i, sample) in neural_data.iter_mut().enumerate().take(num_channels) {
            // Simulate varying spike activity with smaller variations
            // to avoid Rice encoding overflow
            *sample = if i % 100 < spike_density {
                100 + ((i % 20) as i32)
            } else {
                100
            };
        }

        // Test each quality level
        let quality_levels = [
            QualityLevel::Lossless,
            QualityLevel::ReducedPrecision,
            QualityLevel::Lossy4Bit,
            QualityLevel::SummaryOnly,
        ];

        for quality in quality_levels {
            let mut compressed = [0u8; 8192];
            let size =
                compress_adaptive(&neural_data, &mut compressed, &mut workspace, quality).unwrap();

            // Record the compressed size
            controller.record_compressed_size(size);

            // Note: Floating-point arithmetic used here for display purposes only.
            // The QoS module itself uses pure integer arithmetic for no_std compatibility.
            println!(
                "  {:?}: {} bytes ({:.1}% of raw)",
                quality,
                size,
                (size as f64 / (num_channels * 4) as f64) * 100.0
            );

            // Verify roundtrip
            let mut decompressed = [0i32; 1024];
            decompress_adaptive(&compressed[..size], &mut decompressed, &mut workspace).unwrap();
        }

        // Simulate bandwidth estimation at 1000 Hz sample rate
        let (avg_size, estimated_bw) = controller.bandwidth_stats();
        println!(
            "\n  Average size: {} bytes, Estimated BW @ 1kHz: {} bytes/s",
            avg_size, estimated_bw
        );

        // Simulate buffer occupancy
        let buffer_occupancy = match name {
            "Low activity (resting)" => 2000,   // 25% of max
            "Medium activity (movement)" => 5000, // 60% of max
            "High activity (seizure)" => 7500,  // 90% of max
            _ => 2000,
        };
        controller.update_buffer_occupancy(buffer_occupancy);

        // Select appropriate quality based on conditions
        let selected_quality = controller.select_quality(1000);
        // Note: Floating-point used for display only (percentage calculation)
        println!("  Buffer occupancy: {} bytes ({:.0}% full)", 
            buffer_occupancy,
            (buffer_occupancy as f64 / config.max_buffer_bytes as f64) * 100.0
        );
        println!("  Recommended quality: {:?}\n", selected_quality);
    }

    println!("=== Use Case: Wireless Implant ===");
    println!("The QoS controller automatically adjusts quality based on:");
    println!("  1. Available bandwidth (target vs estimated)");
    println!("  2. Buffer occupancy (prevent overflow)");
    println!("  3. Neural activity level (adapt to data characteristics)");
    println!("\nBenefits:");
    println!("  ✓ Prevents data loss during high activity");
    println!("  ✓ Maintains predictable bandwidth usage");
    println!("  ✓ Graceful degradation under constraints");
    println!("  ✓ Automatic recovery when conditions improve");

    println!("\n=== State Machine Behavior ===");
    println!("Lossless → ReducedPrecision (buffer > 80% OR bandwidth exceeded)");
    println!("ReducedPrecision → Lossy4Bit (buffer > 90%)");
    println!("Lossy4Bit → ReducedPrecision → Lossless (buffer < 50% AND bandwidth OK)");
    println!("\nHysteresis prevents rapid oscillation between states.");
}
