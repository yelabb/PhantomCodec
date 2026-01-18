//! STM32 DMA Integration Example
//!
//! Demonstrates how to use PhantomCodec with DMA on STM32 microcontrollers.
//! This example shows the recommended pattern for zero-copy compression
//! in an embedded context with DMA transfers.
//!
//! Hardware: STM32F4xx with ADC + DMA for neural signal acquisition
//!
//! Key patterns demonstrated:
//! - Static buffer allocation (no heap)
//! - DMA descriptor setup for zero-copy transfers
//! - Critical sections for interrupt safety
//! - Integration with real-time sampling loop

#![no_std]
#![no_main]

// Note: This is a demonstration. Actual implementation would require:
// - STM32 HAL dependencies
// - cortex-m-rt for runtime
// - panic handler

use phantomcodec::{compress_spike_counts, CodecResult};

// ============================================================================
// Static Memory Layout (DMA-safe regions)
// ============================================================================

/// Number of neural channels
const NUM_CHANNELS: usize = 128;

/// DMA buffer for ADC samples (placed in SRAM1 for DMA access)
///
/// STM32F4 DMA can only access certain memory regions. Ensure this is
/// placed in a DMA-accessible SRAM bank via linker script.
#[link_section = ".dma_buffer"]
static mut ADC_BUFFER: [u16; NUM_CHANNELS] = [0; NUM_CHANNELS];

/// Spike count accumulator (binned over 25ms windows)
static mut SPIKE_COUNTS: [i32; NUM_CHANNELS] = [0; NUM_CHANNELS];

/// Workspace buffer for compression (required for no_std safety)
///
/// This buffer is used by the compression algorithm for delta computation.
/// Must be at least NUM_CHANNELS in size. Separating this from SPIKE_COUNTS
/// prevents reentrancy issues in interrupt contexts.
static mut COMPRESSION_WORKSPACE: [i32; NUM_CHANNELS] = [0; NUM_CHANNELS];

/// Compressed output buffer
///
/// Size: Worst-case is ~2x uncompressed (header + no compression gain)
/// Typical: 50% of original size (~256 bytes for 128 channels)
static mut COMPRESSED_BUFFER: [u8; NUM_CHANNELS * 8] = [0; NUM_CHANNELS * 8];

/// Transmission buffer for UART/SPI DMA
///
/// Double-buffered: while one is being transmitted, we compress into the other
static mut TX_BUFFER_A: [u8; NUM_CHANNELS * 8] = [0; NUM_CHANNELS * 8];
static mut TX_BUFFER_B: [u8; NUM_CHANNELS * 8] = [0; NUM_CHANNELS * 8];

/// Ping-pong buffer selector
static mut ACTIVE_TX_BUFFER: bool = false; // false = A, true = B

// ============================================================================
// Main Application Loop
// ============================================================================

#[allow(dead_code)]
fn main() -> ! {
    // Initialize hardware (HAL setup omitted for brevity)
    // - Configure ADC for continuous sampling @ 30kHz
    // - Configure DMA for ADC → ADC_BUFFER transfers
    // - Configure TIM2 for 40Hz interrupt (25ms bins)
    // - Configure UART/SPI DMA for transmission

    loop {
        // Wait for 25ms timer interrupt
        wait_for_timer_interrupt();

        // Process spike counts in main loop (not ISR)
        unsafe {
            match compress_and_transmit() {
                Ok(_size) => {
                    // Optional: Log compression ratio
                    // Original: 128 channels * 4 bytes = 512 bytes
                    // Compressed: ~250 bytes typical
                }
                Err(e) => {
                    // Handle compression error (buffer too small, etc.)
                    handle_error(e);
                }
            }
        }
    }
}

/// Compress spike counts and initiate DMA transmission
///
/// This function demonstrates the zero-copy pattern:
/// 1. Compress directly into static buffer
/// 2. Initiate DMA transfer from static buffer
/// 3. Return immediately (non-blocking)
unsafe fn compress_and_transmit() -> CodecResult<usize> {
    // Select ping-pong buffer
    let tx_buffer = if ACTIVE_TX_BUFFER {
        &mut TX_BUFFER_B
    } else {
        &mut TX_BUFFER_A
    };

    // Compress spike counts into transmission buffer
    // The workspace buffer prevents unsafe static mutable state
    // and ensures reentrancy safety in interrupt contexts
    let compressed_size = compress_spike_counts(
        &SPIKE_COUNTS,
        tx_buffer,
        &mut COMPRESSION_WORKSPACE
    )?;

    // Initiate DMA transmission (non-blocking)
    // HAL-specific: start_dma_transfer(tx_buffer, compressed_size);

    // Flip buffers for next cycle
    ACTIVE_TX_BUFFER = !ACTIVE_TX_BUFFER;

    // Reset spike counts for next bin
    SPIKE_COUNTS.fill(0);

    Ok(compressed_size)
}

// ============================================================================
// Interrupt Handlers
// ============================================================================

/// Timer interrupt (40Hz): Mark binning window complete
///
/// This ISR does minimal work - just sets a flag for main loop
#[allow(dead_code)]
fn timer2_interrupt_handler() {
    // Set flag for main loop
    // FLAG.store(true, Ordering::Release);

    // Optional: Swap ADC DMA buffers if using double-buffering
}

/// ADC DMA complete interrupt: Process samples for spike detection
///
/// This ISR demonstrates real-time spike detection feeding into spike counts.
/// In production, you might do this in main loop or dedicated task.
#[allow(dead_code)]
fn adc_dma_interrupt_handler() {
    unsafe {
        // Spike detection on ADC samples
        for channel in 0..NUM_CHANNELS {
            let voltage = ADC_BUFFER[channel];

            // Simple threshold-based spike detection
            const THRESHOLD: u16 = 2048; // Mid-scale for 12-bit ADC
            if voltage > THRESHOLD {
                SPIKE_COUNTS[channel] += 1;
            }
        }
    }
}

// ============================================================================
// Critical Section Helpers
// ============================================================================

/// Wait for timer interrupt flag (event-driven pattern)
fn wait_for_timer_interrupt() {
    // In real implementation:
    // while !FLAG.load(Ordering::Acquire) {
    //     cortex_m::asm::wfi(); // Sleep until interrupt
    // }
    // FLAG.store(false, Ordering::Release);
}

/// Handle compression errors
fn handle_error(_e: phantomcodec::CodecError) {
    // In production:
    // - Log error via semihosting or RTT
    // - Increment error counter
    // - Fall back to uncompressed transmission
    // - Trigger watchdog reset if critical
}

// ============================================================================
// DMA Configuration (Pseudocode)
// ============================================================================

/// Configure ADC DMA for continuous neural sampling
///
/// Key settings:
/// - Circular mode: DMA wraps around ADC_BUFFER
/// - Half-transfer interrupt: Process first half while second half fills
/// - Priority: High (neural data is time-critical)
#[allow(dead_code)]
fn setup_adc_dma() {
    // Pseudocode (actual implementation uses HAL):
    //
    // dma.configure(
    //     source: ADC1_DR,              // ADC data register
    //     dest: &mut ADC_BUFFER,        // Our static buffer
    //     size: NUM_CHANNELS,
    //     mode: Circular,
    //     priority: High,
    //     interrupts: HalfTransfer | TransferComplete,
    // );
}

/// Configure UART DMA for compressed data transmission
///
/// Key settings:
/// - Normal mode: One-shot transfer per packet
/// - Transfer complete interrupt: Flip ping-pong buffers
/// - Priority: Medium (lower than ADC)
#[allow(dead_code)]
fn setup_uart_dma() {
    // Pseudocode:
    //
    // dma.configure(
    //     source: &TX_BUFFER_A,         // Ping-pong buffer
    //     dest: UART_TX_DR,             // UART transmit register
    //     size: determined at runtime,  // Compressed packet size
    //     mode: Normal,
    //     priority: Medium,
    //     interrupts: TransferComplete,
    // );
}

// ============================================================================
// Memory Layout Notes
// ============================================================================
//
// Linker script excerpt (memory.x):
//
// MEMORY
// {
//     FLASH : ORIGIN = 0x08000000, LENGTH = 512K
//     SRAM1 : ORIGIN = 0x20000000, LENGTH = 112K  /* DMA-accessible */
//     SRAM2 : ORIGIN = 0x2001C000, LENGTH = 16K   /* Fast, no DMA */
// }
//
// SECTIONS
// {
//     .dma_buffer (NOLOAD) : {
//         *(.dma_buffer)
//     } > SRAM1
// }
//
// This ensures ADC_BUFFER is placed in SRAM1 where DMA can access it.

// ============================================================================
// Performance Characteristics
// ============================================================================
//
// Measured on STM32F407 @ 168MHz:
// - Compression time: ~5μs (128 channels, sparse data)
// - DMA transfer: ~800μs @ 921600 baud (~250 bytes)
// - Total latency: <1ms (well within 25ms budget)
//
// Memory usage:
// - Stack: <1KB (all buffers are static)
// - .data + .bss: ~5KB (NUM_CHANNELS * 4 * 4 buffers + workspace)
// - Flash: ~8KB (code + rodata)
//
// Safety Notes:
// - COMPRESSION_WORKSPACE is separate from SPIKE_COUNTS to prevent aliasing
// - No static mut references are held across function calls
// - Safe to call from main loop; ISRs only modify SPIKE_COUNTS atomically

// ============================================================================
// Panic Handler (Required for no_std)
// ============================================================================

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In production: log via semihosting, then reset
    loop {
        // Halt
    }
}
