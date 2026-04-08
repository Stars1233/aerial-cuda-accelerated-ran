# Performance Metrics Library

A high-performance timing and metrics collection library for cuPHY modules, designed for zero-overhead performance measurement with NVLOG integration.

## Features

- **Zero Dynamic Allocation**: No memory allocation during timing operations (only during initialization)
- **High-Precision Timing**: Uses `std::chrono::system_clock` for nanosecond-precision measurements
- **NVLOG Integration**: Direct integration with NVLOG system for structured logging
- **Compile-Time Safety**: Template-based interface with compile-time NVLOG tag validation
- **Single-Threaded Design**: Optimized for single-threaded usage (not thread-safe)
- **Minimal Interface**: Only 4 public methods for simplicity

## Quick Start

```cpp
#include "perf_metrics/perf_metrics_accumulator.hpp"

// Define your NVLOG tag (must be compile-time constant)
#define TAG_MY_PERF 54  // Example: NVIPC.TIMING

// Create accumulator with pre-registered sections
perf_metrics::PerfMetricsAccumulator pma{"Section 1", "Section 2"};

// Time your code sections
for (int i = 0; i < 100; i++) {
    pma.startSection("Section 1");
    // ... your code here ...
    pma.stopSection("Section 1");
    
    pma.startSection("Section 2");
    // ... your code here ...
    pma.stopSection("Section 2");
}

// Log accumulated results
pma.logDurations<TAG_MY_PERF>();  // Uses INFO level by default
pma.reset();  // Clear for next measurement cycle
```

## API Reference

### Constructor

```cpp
// Pre-register sections using initializer list
PerfMetricsAccumulator(const std::initializer_list<const char*>& sectionNames);

// Default constructor (sections must be pre-registered)
PerfMetricsAccumulator();
```

### Core Methods

```cpp
// Start timing a section
void startSection(const char* sectionName);

// Stop timing a section
void stopSection(const char* sectionName);

// Log accumulated durations with specified NVLOG tag and level
template<int NvlogTag, LogLevel Level = LogLevel::INFO>
void logDurations() const;

// Reset all accumulated data to zero
void reset();
```

### Log Levels

```cpp
enum class LogLevel {
    VERBOSE,  // NVLOGV_FMT
    DEBUG,    // NVLOGD_FMT  
    INFO,     // NVLOGI_FMT (default)
    WARN,     // NVLOGW_FMT
    ERROR     // NVLOGE_FMT (uses AERIAL_CUPHY_EVENT)
};
```

## Usage Examples

### Basic Usage

```cpp
#include "perf_metrics/perf_metrics_accumulator.hpp"

// Define NVLOG tags for your module
#define NVLOG_TAG_BASE_MY_MODULE 1300
#define TAG_MY_PERF_METRICS (NVLOG_TAG_BASE_MY_MODULE + 1)

void example_usage() {
    // IMPORTANT: Pre-register ALL sections to avoid dynamic allocation
    perf_metrics::PerfMetricsAccumulator pma{"Section 1", "Section 2"};
    
    // Simulate work with timing
    for (int ii = 0; ii < 100; ii++) {
        pma.startSection("Section 1");  // Must be pre-registered
        // ... code for Section 1 ...
        pma.stopSection("Section 1");
        
        pma.startSection("Section 2");  // Must be pre-registered  
        // ... code for Section 2 ...
        pma.stopSection("Section 2");
    }
    
    // Log results with different levels
    pma.logDurations<TAG_MY_PERF_METRICS>();                                  // INFO level (default)
    pma.logDurations<TAG_MY_PERF_METRICS, perf_metrics::LogLevel::DEBUG>();  // DEBUG level
    pma.logDurations<TAG_MY_PERF_METRICS, perf_metrics::LogLevel::VERBOSE>(); // VERBOSE level
    
    // Reset for next measurement cycle
    pma.reset();
}
```

### RU Emulator Example

```cpp
#define TAG_RU_PERF (NVLOG_TAG_BASE_RU_EMULATOR + 15)

void ru_emulator_example() {
    perf_metrics::PerfMetricsAccumulator pma{"UL Processing", "DL Processing", "BFW Generation"};
    
    // Time UL processing
    pma.startSection("UL Processing");
    // ... UL processing code ...
    pma.stopSection("UL Processing");
    
    // Time DL processing  
    pma.startSection("DL Processing");
    // ... DL processing code ...
    pma.stopSection("DL Processing");
    
    // Time BFW generation
    pma.startSection("BFW Generation");
    // ... BFW generation code ...
    pma.stopSection("BFW Generation");
    
    // Log with ru-emulator specific tag
    pma.logDurations<TAG_RU_PERF, perf_metrics::LogLevel::INFO>();
}
```

### Clean Usage with Namespace

```cpp
void clean_usage_example() {
    using namespace perf_metrics;
    
    PerfMetricsAccumulator pma{"Section 1", "Section 2"};
    
    // Only 4 methods available: startSection, stopSection, logDurations, reset
    pma.startSection("Section 1");
    // ... timing code ...
    pma.stopSection("Section 1");
    
    pma.logDurations<TAG_MY_PERF_METRICS>();                    // INFO level (default)
    pma.logDurations<TAG_MY_PERF_METRICS, LogLevel::DEBUG>();  // DEBUG level
    
    pma.reset();  // Clear all accumulated data
}
```

## Output Format

The library outputs timing data in a compact, single-line format:

```
PerfMetricsAccumulator - Section1:100:1542,Section2:50:1700
```

Where:
- `Section1:100:1542` means "Section1" was called 100 times and took 1542 microseconds total
- `Section2:50:1700` means "Section2" was called 50 times and took 1700 microseconds total  
- Multiple sections are comma-separated
- Format: `SectionName:count:duration_us`
- Times are reported in microseconds (μs)
- Internal precision is nanoseconds for accuracy

## Error Handling

The library uses `printf` for error reporting:

```cpp
// Starting non-existent section
pma.startSection("Unknown");
// Output: PerfMetricsAccumulator: Error - section 'Unknown' not found. Pre-register all sections in constructor.

// Double start
pma.startSection("Section1");
pma.startSection("Section1");  
// Output: PerfMetricsAccumulator: Error - timing already active for section 'Section1'

// Stop without start
pma.stopSection("Section1");
// Output: PerfMetricsAccumulator: Error - timing not active for section 'Section1'
```

## Performance Characteristics

- **Zero allocation during timing**: All memory allocated at construction time
- **Fixed-size logging buffer**: 1024 bytes, no dynamic allocation for log formatting
- **Nanosecond precision**: Uses `std::chrono::system_clock` for high-precision timing
- **Microsecond output**: Converted for readability while maintaining precision
- **Compile-time optimization**: Template-based logging resolves at compile time

## Integration

### CMake

The library is automatically available when linking with `perf_metrics`:

```cmake
target_link_libraries(your_target PRIVATE perf_metrics)
```

### Dependencies

- **nvlog**: For structured logging output
- **C++20**: Uses modern C++ features like `if constexpr`

## Thread Safety

Currently designed for single-threaded usage. Each thread should have its own `PerfMetricsAccumulator` instance if multi-threading is needed.

## Best Practices

1. **Pre-register all sections** in the constructor to avoid runtime allocation
2. **Use meaningful section names** that clearly identify the code being timed
3. **Choose appropriate NVLOG tags** that follow your module's tag conventions
4. **Reset regularly** to avoid overflow of accumulated values
5. **Use consistent log levels** for similar types of measurements
6. **Keep section names short** to fit within the 1024-byte log buffer

## Testing

Run the unit tests to verify functionality:

```bash
# PerfMetricsAccumulator test
ninja perf_metrics_accumulator_unit_test
./cuPHY-CP/gt_common_libs/perf_metrics/test/perf_metrics_accumulator_unit_test

# PercentileTracker test
ninja percentile_tracker_unit_test
./cuPHY-CP/gt_common_libs/perf_metrics/test/percentile_tracker_unit_test
```

The PerfMetricsAccumulator test covers:
- Basic timing functionality
- Error handling scenarios  
- Different log levels
- Reset functionality
- NVLOG integration

The PercentileTracker test covers:
- Basic statistics (min, max, mean, std deviation)
- Percentile calculations (p50, p90, p99, p99.9, p99.99)
- Single-slot and multi-slot modes
- Reset functionality
- Edge case handling
- Move semantics

---

# PercentileTracker

A histogram-based percentile tracking class for efficient statistical analysis with support for multi-slot tracking.

## Features

- **Histogram-based Percentile Calculation**: Efficient percentile computation using configurable bucket sizes
- **Multi-slot Support**: Track statistics independently for up to 80+ slots (e.g., per 5G NR slot)
- **Single-slot Mode**: Simplified usage for aggregate statistics
- **Comprehensive Statistics**: Min, max, mean, standard deviation, and multiple percentiles (p50, p90, p99, p99.9, p99.99)
- **NVLOG Integration**: Direct logging with configurable log levels
- **Move Semantics**: Efficient object transfer with C++11 move support
- **Memory Tracking**: Built-in memory footprint logging

## Quick Start

```cpp
#include "perf_metrics/percentile_tracker.hpp"

// Define your NVLOG tag
#define TAG_MY_PERF 54  // Example: NVIPC.TIMING

// Create tracker in single-slot mode
// Parameters: lowestValue, highestValue, bucketSize
perf_metrics::PercentileTracker tracker(0, 500000, 1000);
// Tracks 0-500μs range with 1μs bucket resolution

// Add values
for (int i = 0; i < 100; i++) {
    tracker.addValue(processingTime); // Time in nanoseconds
}

// Get statistics
int64_t p50 = tracker.getPercentile(50.0);
int64_t p99 = tracker.getPercentile(99.0);
int64_t mean = tracker.getMean();

// Or get all statistics at once
auto stats = tracker.getStatistics();

// Log the statistics
tracker.logStats<TAG_MY_PERF>(2, "Processing Times");

// Reset for next measurement window
tracker.reset();
```

## API Reference

### Constructor

```cpp
PercentileTracker(int64_t lowestTrackableValue = 0,
                  int64_t highestTrackableValue = 500000,  // 500μs in ns
                  int64_t bucketSize = 1000,               // 1μs buckets
                  int32_t numSlots = -1);                  // -1 = single-slot mode
```

**Parameters:**
- `lowestTrackableValue`: Minimum value to track (nanoseconds)
- `highestTrackableValue`: Maximum value to track (nanoseconds)
- `bucketSize`: Bucket width for histogram (nanoseconds)
- `numSlots`: Number of slots (-1 or 1 for single-slot mode, >1 for multi-slot)

### Adding Values

```cpp
// Single-slot mode
void addValue(int64_t value);

// Multi-slot mode with SFN/slot addressing
void addValue(int64_t value, uint16_t sfn, uint16_t slot);

// Multi-slot mode with direct slot index
void addValue(int64_t value, int slotIndex);
```

### Statistics Methods

```cpp
// Individual statistics
int64_t getMinValue(int slotIndex = -1) const;
int64_t getMaxValue(int slotIndex = -1) const;
int64_t getMean(int slotIndex = -1) const;
int64_t getStdDeviation(int slotIndex = -1) const;
int64_t getTotalCount(int slotIndex = -1) const;

// Percentiles
int64_t getPercentile(double percentile, int slotIndex = -1);

// All statistics in one bundle
Statistics getStatistics(int slotIndex = -1) const;
```

**Note:** `slotIndex = -1` returns aggregate statistics across all slots

### Statistics Structure

```cpp
struct Statistics {
    int64_t totalCount;
    int64_t min;
    int64_t max;
    int64_t mean;
    int64_t stdDev;
    int64_t p50;
    int64_t p90;
    int64_t p99;
    int64_t p999;
    int64_t p9999;
};
```

### Logging Methods

```cpp
// Log statistics with configurable level
// Levels: 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG
template<int PT_LOGGING_TAG>
void logStats(uint32_t logLevel = 2, const char* prefix = "", int slotIndex = -1) const;

// Log memory usage
template<int PT_LOGGING_TAG>
void logMemoryFootprint(uint32_t logLevel = 2) const;
```

### Utility Methods

```cpp
// Reset statistics
void reset(int slotIndex = -1);  // -1 resets all slots

// Convert SFN/slot to slot index
uint32_t getSlotIndex(uint16_t sfn, uint16_t slot) const;
```

## Usage Examples

### Single-Slot Mode (Aggregate Statistics)

```cpp
using namespace perf_metrics;

// Track latency from 0-1ms with 10μs resolution
PercentileTracker latency(0, 1000000, 10000);

// Add measurements
for (int i = 0; i < 1000; i++) {
    int64_t startTime = getCurrentTime();
    processData();
    int64_t duration = getCurrentTime() - startTime;
    latency.addValue(duration);
}

// Get statistics
auto stats = latency.getStatistics();
std::cout << "Mean: " << stats.mean << "ns" << std::endl;
std::cout << "P99: " << stats.p99 << "ns" << std::endl;

// Log with custom prefix
latency.logStats<TAG_MY_PERF>(2, "Data Processing");
```

### Multi-Slot Mode (Per-Slot Statistics)

```cpp
// Track statistics for 20 slots (one 5G NR frame)
PercentileTracker slotLatency(0, 500000, 1000, 20);

// Add values with SFN/slot addressing
for (uint16_t sfn = 0; sfn < 10; sfn++) {
    for (uint16_t slot = 0; slot < 20; slot++) {
        int64_t processingTime = measureSlotProcessing(sfn, slot);
        slotLatency.addValue(processingTime, sfn, slot);
    }
}

// Get statistics for specific slot
auto slot0Stats = slotLatency.getStatistics(0);
slotLatency.logStats<TAG_MY_PERF>(2, "Slot 0", 0);

// Get aggregate statistics across all slots
auto allStats = slotLatency.getStatistics(-1);
slotLatency.logStats<TAG_MY_PERF>(2, "All Slots", -1);

// Reset specific slot
slotLatency.reset(5);

// Reset all slots
slotLatency.reset(-1);
```

### Using Direct Slot Index

```cpp
PercentileTracker tracker(0, 100000, 1000, 10);

// Add values directly by slot index
for (int slotIdx = 0; slotIdx < 10; slotIdx++) {
    for (int i = 0; i < 100; i++) {
        tracker.addValue(measureLatency(slotIdx), slotIdx);
    }
}

// Query each slot independently
for (int slotIdx = 0; slotIdx < 10; slotIdx++) {
    int64_t p99 = tracker.getPercentile(99.0, slotIdx);
    std::cout << "Slot " << slotIdx << " P99: " << p99 << std::endl;
}
```

## Output Format

The `logStats` method outputs statistics in a compact, fixed-width format:

```
Processing Times          | slot=04 count=1000          | min=1234    max=98765   mean=45678  std=12345   | p50=44000   p90=76000   p99=91000   p99.9=96000   p99.99=98000
Processing Times          | slot=all count=5000          | min=1234    max=98765   mean=45678  std=12345   | p50=44000   p90=76000   p99=91000   p99.9=96000   p99.99=98000
```

Where all values are in nanoseconds.

## Performance Characteristics

- **Memory Usage**: `sizeof(PercentileTracker) + (numSlots * bucketCount * 8 bytes)`
  - Example: 500 buckets × 20 slots = 80KB
- **addValue() Complexity**: O(1) - constant time bucket update
- **getPercentile() Complexity**: O(bucketCount) - linear scan through histogram
- **Memory Allocation**: All memory allocated at construction time
- **Thread Safety**: Not thread-safe; use one instance per thread

## Best Practices

1. **Choose appropriate bucket size**: Balance between precision and memory usage
   - 1μs buckets for sub-millisecond measurements
   - 10μs buckets for millisecond-range measurements
   
2. **Set realistic ranges**: Avoid excessive bucket counts
   - Maximum 100K buckets per slot (enforced by constructor)
   
3. **Use single-slot mode when possible**: Lower memory overhead
   
4. **Reset regularly**: Prevents overflow and provides windowed statistics
   
5. **Use Statistics bundle**: More efficient than individual method calls
   
6. **Log memory footprint**: Monitor memory usage during development

## Memory Footprint Examples

```cpp
// Small: 100 buckets × 1 slot ≈ 1KB
PercentileTracker small(0, 100000, 1000, 1);

// Medium: 500 buckets × 20 slots ≈ 80KB
PercentileTracker medium(0, 500000, 1000, 20);

// Large: 1000 buckets × 80 slots ≈ 640KB
PercentileTracker large(0, 1000000, 1000, 80);

// Check actual memory usage
small.logMemoryFootprint<TAG_MY_PERF>();
```

## Integration

The PercentileTracker is automatically available when linking with `perf_metrics`:

```cmake
target_link_libraries(your_target PRIVATE perf_metrics)
```

### Dependencies

- **nvlog**: For structured logging output
- **C++20**: Uses modern C++ features
