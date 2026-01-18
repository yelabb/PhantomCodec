//! Quality of Service (QoS) auto-throttling for adaptive bandwidth management
//!
//! This module provides dynamic compression strategy selection based on bandwidth
//! constraints and buffer occupancy. It automatically switches between lossless
//! and lossy compression to maintain target bandwidth while preserving data fidelity.

/// QoS configuration for adaptive compression
#[derive(Debug, Clone, Copy)]
pub struct QosConfig {
    /// Target output bandwidth in bytes per second
    pub target_bandwidth: u32,

    /// Maximum buffer size in bytes before aggressive compression
    pub max_buffer_bytes: usize,

    /// Priority mode for compression decisions
    pub priority: QosPriority,

    /// Minimum quality level (never go below this)
    pub min_quality: QualityLevel,
}

impl QosConfig {
    /// Create a new QoS configuration with sensible defaults
    ///
    /// # Arguments
    /// * `target_bandwidth` - Target bandwidth in bytes per second
    ///
    /// # Example
    /// ```
    /// use phantomcodec::qos::{QosConfig, QosPriority, QualityLevel};
    ///
    /// let config = QosConfig::new(100_000); // 100 KB/s target
    /// assert_eq!(config.target_bandwidth, 100_000);
    /// ```
    pub const fn new(target_bandwidth: u32) -> Self {
        Self {
            target_bandwidth,
            max_buffer_bytes: 8192, // 8 KB default
            priority: QosPriority::Balanced,
            min_quality: QualityLevel::Lossy4Bit,
        }
    }

    /// Create a QoS configuration optimized for spike preservation
    pub const fn preserve_spikes(target_bandwidth: u32) -> Self {
        Self {
            target_bandwidth,
            max_buffer_bytes: 8192,
            priority: QosPriority::PreserveSpikes,
            min_quality: QualityLevel::ReducedPrecision,
        }
    }

    /// Create a QoS configuration optimized for LFP preservation
    pub const fn preserve_lfp(target_bandwidth: u32) -> Self {
        Self {
            target_bandwidth,
            max_buffer_bytes: 8192,
            priority: QosPriority::PreserveLfp,
            min_quality: QualityLevel::Lossy4Bit,
        }
    }
}

/// Priority mode for QoS decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QosPriority {
    /// Preserve spike timing at all costs (prefer lossless)
    PreserveSpikes = 0,
    /// Preserve LFP shape (for oscillation analysis)
    PreserveLfp = 1,
    /// Balanced approach (default)
    Balanced = 2,
}

/// Quality level for compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum QualityLevel {
    /// Lossless compression (Rice/FixedWidth)
    Lossless = 0,
    /// Reduced precision (10-bit quantization)
    ReducedPrecision = 1,
    /// Lossy 4-bit packing
    Lossy4Bit = 2,
    /// Summary only (spike counts, no waveforms)
    SummaryOnly = 3,
}

/// Bandwidth monitor for tracking rolling output size
#[derive(Debug, Clone)]
pub struct BandwidthMonitor {
    /// Rolling window of compressed sizes (bytes)
    window: [usize; 16],
    /// Current position in the window
    position: usize,
    /// Number of samples in the window
    count: usize,
    /// Total bytes across all samples in window
    total_bytes: usize,
}

impl BandwidthMonitor {
    /// Create a new bandwidth monitor
    pub const fn new() -> Self {
        Self {
            window: [0; 16],
            position: 0,
            count: 0,
            total_bytes: 0,
        }
    }

    /// Record a new compressed size sample
    ///
    /// # Arguments
    /// * `size` - Compressed size in bytes
    pub fn record(&mut self, size: usize) {
        // Remove old value from total if window is full
        if self.count == self.window.len() {
            self.total_bytes -= self.window[self.position];
        } else {
            self.count += 1;
        }

        // Add new value
        self.window[self.position] = size;
        self.total_bytes += size;

        // Advance position
        self.position = (self.position + 1) % self.window.len();
    }

    /// Get average bytes per sample
    pub fn average_size(&self) -> usize {
        if self.count == 0 {
            0
        } else {
            self.total_bytes / self.count
        }
    }

    /// Estimate bandwidth in bytes per second
    ///
    /// # Arguments
    /// * `sample_rate_hz` - Sampling rate in Hz
    pub fn estimate_bandwidth(&self, sample_rate_hz: u32) -> u32 {
        let avg = self.average_size() as u32;
        avg.saturating_mul(sample_rate_hz)
    }

    /// Reset the monitor
    pub fn reset(&mut self) {
        self.window = [0; 16];
        self.position = 0;
        self.count = 0;
        self.total_bytes = 0;
    }
}

impl Default for BandwidthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// QoS Controller state machine
#[derive(Debug, Clone)]
pub struct QosController {
    /// Current quality level
    current_quality: QualityLevel,
    /// QoS configuration
    config: QosConfig,
    /// Bandwidth monitor
    bandwidth_monitor: BandwidthMonitor,
    /// Current buffer occupancy (bytes)
    buffer_occupancy: usize,
    /// Hysteresis counter to prevent oscillation
    hysteresis_counter: u8,
}

impl QosController {
    /// Create a new QoS controller
    pub fn new(config: QosConfig) -> Self {
        Self {
            current_quality: QualityLevel::Lossless,
            config,
            bandwidth_monitor: BandwidthMonitor::new(),
            buffer_occupancy: 0,
            hysteresis_counter: 0,
        }
    }

    /// Get the current quality level
    pub fn current_quality(&self) -> QualityLevel {
        self.current_quality
    }

    /// Update buffer occupancy
    pub fn update_buffer_occupancy(&mut self, occupancy: usize) {
        self.buffer_occupancy = occupancy;
    }

    /// Record a compressed size sample
    pub fn record_compressed_size(&mut self, size: usize) {
        self.bandwidth_monitor.record(size);
    }

    /// Get bandwidth statistics
    pub fn bandwidth_stats(&self) -> (usize, u32) {
        (
            self.bandwidth_monitor.average_size(),
            self.bandwidth_monitor.estimate_bandwidth(1000), // Assume 1kHz default
        )
    }

    /// Select the appropriate quality level based on current conditions
    ///
    /// # Arguments
    /// * `sample_rate_hz` - Sampling rate in Hz for bandwidth estimation
    ///
    /// # Returns
    /// The selected quality level
    pub fn select_quality(&mut self, sample_rate_hz: u32) -> QualityLevel {
        let buffer_ratio = (self.buffer_occupancy as f32) / (self.config.max_buffer_bytes as f32);
        let estimated_bw = self.bandwidth_monitor.estimate_bandwidth(sample_rate_hz);

        // Check if we need to degrade quality
        let should_degrade = buffer_ratio > 0.8 || estimated_bw > self.config.target_bandwidth;

        // Check if we can improve quality
        let should_improve = buffer_ratio < 0.5 && estimated_bw < self.config.target_bandwidth;

        // Apply hysteresis to prevent oscillation
        if should_degrade && self.current_quality < self.config.min_quality {
            self.hysteresis_counter += 1;
            if self.hysteresis_counter >= 3 {
                // Degrade quality level
                self.current_quality = match self.current_quality {
                    QualityLevel::Lossless => QualityLevel::ReducedPrecision,
                    QualityLevel::ReducedPrecision => QualityLevel::Lossy4Bit,
                    QualityLevel::Lossy4Bit => QualityLevel::SummaryOnly,
                    QualityLevel::SummaryOnly => QualityLevel::SummaryOnly,
                };
                // Ensure we don't go below minimum quality
                if self.current_quality > self.config.min_quality {
                    self.current_quality = self.config.min_quality;
                }
                self.hysteresis_counter = 0;
            }
        } else if should_improve && self.current_quality > QualityLevel::Lossless {
            self.hysteresis_counter += 1;
            if self.hysteresis_counter >= 5 {
                // Improve quality level
                self.current_quality = match self.current_quality {
                    QualityLevel::SummaryOnly => QualityLevel::Lossy4Bit,
                    QualityLevel::Lossy4Bit => QualityLevel::ReducedPrecision,
                    QualityLevel::ReducedPrecision => QualityLevel::Lossless,
                    QualityLevel::Lossless => QualityLevel::Lossless,
                };
                self.hysteresis_counter = 0;
            }
        } else {
            // Reset hysteresis counter if conditions are stable
            self.hysteresis_counter = 0;
        }

        self.current_quality
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.current_quality = QualityLevel::Lossless;
        self.bandwidth_monitor.reset();
        self.buffer_occupancy = 0;
        self.hysteresis_counter = 0;
    }
}

/// Statistics about QoS operation
#[derive(Debug, Clone, Copy)]
pub struct QosStats {
    /// Current quality level
    pub current_quality: QualityLevel,
    /// Average compressed size in bytes
    pub average_size: usize,
    /// Estimated bandwidth in bytes per second
    pub estimated_bandwidth: u32,
    /// Current buffer occupancy in bytes
    pub buffer_occupancy: usize,
}

/// Estimate the bandwidth required for given input data
///
/// This is a heuristic based on data characteristics.
///
/// # Arguments
/// * `samples` - Input samples
///
/// # Returns
/// Estimated compressed size in bytes
pub fn estimate_bandwidth(samples: &[i32]) -> usize {
    // Simple heuristic: estimate based on number of samples
    // Assume average compression ratio of ~60% for typical neural data
    let raw_size = samples.len() * core::mem::size_of::<i32>();
    (raw_size as f32 * 0.6) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qos_config_new() {
        let config = QosConfig::new(100_000);
        assert_eq!(config.target_bandwidth, 100_000);
        assert_eq!(config.max_buffer_bytes, 8192);
        assert_eq!(config.priority, QosPriority::Balanced);
        assert_eq!(config.min_quality, QualityLevel::Lossy4Bit);
    }

    #[test]
    fn test_qos_config_preserve_spikes() {
        let config = QosConfig::preserve_spikes(100_000);
        assert_eq!(config.priority, QosPriority::PreserveSpikes);
        assert_eq!(config.min_quality, QualityLevel::ReducedPrecision);
    }

    #[test]
    fn test_bandwidth_monitor() {
        let mut monitor = BandwidthMonitor::new();
        assert_eq!(monitor.average_size(), 0);

        // Record some samples
        monitor.record(100);
        monitor.record(200);
        monitor.record(150);

        assert_eq!(monitor.average_size(), 150); // (100 + 200 + 150) / 3
    }

    #[test]
    fn test_bandwidth_monitor_rolling_window() {
        let mut monitor = BandwidthMonitor::new();

        // Fill the window with values 100..116
        for i in 0..16 {
            monitor.record(100 + i);
        }

        assert_eq!(monitor.count, 16);

        // Add one more, should replace oldest (100)
        monitor.record(200);
        assert_eq!(monitor.count, 16);

        // Average should include 101..115 (15 values) plus 200
        // Window now contains: [200, 101, 102, ..., 115]
        let sum: usize = (101..=115).sum::<usize>() + 200;
        assert_eq!(monitor.average_size(), sum / 16);
    }

    #[test]
    fn test_bandwidth_estimation() {
        let mut monitor = BandwidthMonitor::new();
        monitor.record(100);

        // At 1000 Hz sample rate, bandwidth = 100 * 1000 = 100,000 bytes/s
        assert_eq!(monitor.estimate_bandwidth(1000), 100_000);
    }

    #[test]
    fn test_qos_controller_degradation() {
        let config = QosConfig::new(50_000);
        let mut controller = QosController::new(config);

        // Start at lossless
        assert_eq!(controller.current_quality(), QualityLevel::Lossless);

        // Simulate high buffer occupancy
        controller.update_buffer_occupancy(7000); // 85% of 8192

        // Should trigger degradation after hysteresis
        for _ in 0..3 {
            controller.select_quality(1000);
        }

        // Should have degraded
        assert!(controller.current_quality() > QualityLevel::Lossless);
    }

    #[test]
    fn test_qos_controller_improvement() {
        let config = QosConfig::new(100_000);
        let mut controller = QosController::new(config);

        // Start at reduced precision
        controller.current_quality = QualityLevel::ReducedPrecision;

        // Simulate low buffer occupancy
        controller.update_buffer_occupancy(1000); // ~12% of 8192
        controller.record_compressed_size(50); // Low bandwidth usage

        // Should trigger improvement after hysteresis
        for _ in 0..5 {
            controller.select_quality(1000);
        }

        // Should have improved
        assert_eq!(controller.current_quality(), QualityLevel::Lossless);
    }

    #[test]
    fn test_quality_level_ordering() {
        assert!(QualityLevel::Lossless < QualityLevel::ReducedPrecision);
        assert!(QualityLevel::ReducedPrecision < QualityLevel::Lossy4Bit);
        assert!(QualityLevel::Lossy4Bit < QualityLevel::SummaryOnly);
    }

    #[test]
    fn test_estimate_bandwidth() {
        let samples = [0i32; 1024];
        let estimate = estimate_bandwidth(&samples);

        // Should be roughly 60% of raw size
        let raw_size = 1024 * 4;
        let expected = (raw_size as f32 * 0.6) as usize;
        assert_eq!(estimate, expected);
    }

    #[test]
    fn test_bandwidth_monitor_reset() {
        let mut monitor = BandwidthMonitor::new();
        monitor.record(100);
        monitor.record(200);

        monitor.reset();

        assert_eq!(monitor.count, 0);
        assert_eq!(monitor.average_size(), 0);
    }

    #[test]
    fn test_qos_controller_respects_min_quality() {
        let config = QosConfig {
            target_bandwidth: 1000, // Very low to force degradation
            max_buffer_bytes: 8192,
            priority: QosPriority::Balanced,
            min_quality: QualityLevel::ReducedPrecision,
        };
        let mut controller = QosController::new(config);

        // Simulate high buffer occupancy
        controller.update_buffer_occupancy(8000);

        // Try to degrade many times
        for _ in 0..20 {
            controller.select_quality(1000);
        }

        // Should not go below min_quality
        assert!(controller.current_quality() <= QualityLevel::ReducedPrecision);
    }
}
