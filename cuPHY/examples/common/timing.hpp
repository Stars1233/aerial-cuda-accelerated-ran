/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if !defined(CUPHY_EXAMPLES_TIMING_HPP_INCLUDED_)
#define CUPHY_EXAMPLES_TIMING_HPP_INCLUDED_

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cuphy::examples::timing
{

/** Unit associated with samples in a timing series. */
enum class TimeUnit
{
    Microseconds, ///< Elapsed time measured in microseconds.
    Count         ///< Dimensionless event count.
};

/**
 * Returns the JSON and console label for a sample unit.
 *
 * @param[in] unit Sample unit to label.
 * @return Stable short unit label.
 */
inline const char* unit_string(TimeUnit unit)
{
    switch(unit)
    {
    case TimeUnit::Microseconds:
        return "us";
    case TimeUnit::Count:
        return "count";
    }
    return "unknown";
}

/** Measures elapsed host time with a monotonic clock. */
class CpuTimer
{
public:
    /** Starts a new host timing interval. */
    void start()
    {
        start_time_ = clock_t::now();
        running_    = true;
    }

    /**
     * Stops the active interval and returns its duration.
     *
     * @return Elapsed host time in microseconds.
     * @throws std::logic_error if no interval is active.
     */
    double stop_us()
    {
        if(!running_)
        {
            throw std::logic_error("CpuTimer::stop_us called before start");
        }
        const auto stop_time = clock_t::now();
        running_             = false;
        return std::chrono::duration<double, std::micro>(stop_time - start_time_).count();
    }

private:
    using clock_t = std::chrono::steady_clock;

    clock_t::time_point start_time_{};
    bool                running_ = false;
};

/** Measures elapsed GPU time between events recorded on a CUDA stream. */
class GpuEventTimer
{
public:
    /**
     * Creates the CUDA events used for timing.
     *
     * @throws std::runtime_error if CUDA event creation fails.
     */
    GpuEventTimer()
    {
        check_cuda(cudaEventCreateWithFlags(&begin_event_, cudaEventDefault), "cudaEventCreateWithFlags");
        check_cuda(cudaEventCreateWithFlags(&end_event_, cudaEventDefault), "cudaEventCreateWithFlags");
    }

    /** Destroys the CUDA events owned by this timer. */
    ~GpuEventTimer()
    {
        if(begin_event_ != nullptr)
        {
            static_cast<void>(cudaEventDestroy(begin_event_));
        }
        if(end_event_ != nullptr)
        {
            static_cast<void>(cudaEventDestroy(end_event_));
        }
    }

    /** Copying CUDA event ownership is not supported. */
    GpuEventTimer(const GpuEventTimer&)            = delete;
    /** Copying CUDA event ownership is not supported. */
    GpuEventTimer& operator=(const GpuEventTimer&) = delete;

    /**
     * Records the start event on a CUDA stream.
     *
     * @param[in] stream CUDA stream executing the measured work.
     * @throws std::runtime_error if CUDA event recording fails.
     */
    void record_begin(cudaStream_t stream)
    {
        check_cuda(cudaEventRecord(begin_event_, stream), "cudaEventRecord");
        recorded_begin_ = true;
        recorded_end_   = false;
    }

    /**
     * Records the end event on a CUDA stream.
     *
     * @param[in] stream CUDA stream executing the measured work.
     * @throws std::logic_error if @ref record_begin has not been called.
     * @throws std::runtime_error if CUDA event recording fails.
     */
    void record_end(cudaStream_t stream)
    {
        if(!recorded_begin_)
        {
            throw std::logic_error("GpuEventTimer::record_end called before record_begin");
        }
        check_cuda(cudaEventRecord(end_event_, stream), "cudaEventRecord");
        recorded_end_ = true;
    }

    /**
     * Waits for the recorded end event to complete.
     *
     * @throws std::logic_error if @ref record_end has not been called.
     * @throws std::runtime_error if CUDA event synchronization fails.
     */
    void synchronize()
    {
        if(!recorded_end_)
        {
            throw std::logic_error("GpuEventTimer::synchronize called before record_end");
        }
        check_cuda(cudaEventSynchronize(end_event_), "cudaEventSynchronize");
    }

    /**
     * Returns the elapsed time between the recorded events.
     *
     * @return Elapsed GPU time in microseconds.
     * @throws std::logic_error if @ref record_end has not been called.
     * @throws std::runtime_error if CUDA event timing fails.
     */
    double elapsed_us() const
    {
        if(!recorded_end_)
        {
            throw std::logic_error("GpuEventTimer::elapsed_us called before record_end");
        }
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, begin_event_, end_event_), "cudaEventElapsedTime");
        return static_cast<double>(elapsed_ms) * 1000.0;
    }

private:
    static void check_cuda(cudaError_t status, const char* api)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(std::string(api) + " failed: " + cudaGetErrorString(status));
        }
    }

    cudaEvent_t begin_event_ = nullptr;
    cudaEvent_t end_event_   = nullptr;
    bool        recorded_begin_ = false;
    bool        recorded_end_   = false;
};

/** Descriptive statistics calculated from a sample series. */
struct SummaryStats
{
    size_t                count = 0; ///< Number of samples summarized.
    std::optional<double> min;       ///< Minimum sample value, or empty for no samples.
    std::optional<double> max;       ///< Maximum sample value, or empty for no samples.
    std::optional<double> mean;      ///< Arithmetic mean, or empty for no samples.
    std::optional<double> stddev;    ///< Sample standard deviation, or empty for no samples.
    std::optional<double> p50;       ///< 50th-percentile value, or empty for no samples.
    std::optional<double> p90;       ///< 90th-percentile value, or empty for no samples.
    std::optional<double> p95;       ///< 95th-percentile value, or empty for no samples.
    std::optional<double> p99;       ///< 99th-percentile value, or empty for no samples.
};

/** Specifies the range and resolution of a generated histogram. */
struct HistogramConfig
{
    size_t                bin_count = 50; ///< Number of bins when @ref bin_width is unset.
    std::optional<double> min;            ///< Lower range bound; samples below it are underflow.
    std::optional<double> max;            ///< Upper range bound; samples above it are overflow.
    std::optional<double> bin_width;      ///< Requested bin width; overrides @ref bin_count.
};

/** One half-open histogram interval, except the final upper bound is inclusive. */
struct HistogramBin
{
    double lower = 0.0; ///< Inclusive lower interval bound.
    double upper = 0.0; ///< Exclusive upper bound, except for the final bin.
    size_t count = 0;   ///< Number of samples assigned to this interval.
};

/** Histogram data generated from a sample series. */
struct Histogram
{
    std::vector<HistogramBin> bins;          ///< Ordered intervals spanning the configured range.
    size_t                    underflow = 0; ///< Samples below the configured range.
    size_t                    overflow  = 0; ///< Samples above the configured range.
};

/** Formatting controls for one-line summary output. */
struct ConsoleSummaryOptions
{
    int precision = 3; ///< Decimal digits emitted for numeric values.
};

/** Formatting controls for tabular phase summaries. */
struct ConsoleTableOptions
{
    int    precision     = 3;    ///< Decimal digits emitted for numeric values.
    size_t label_width   = 14;   ///< Width of the phase-label column.
    size_t numeric_width = 10;   ///< Width of each numeric column.
    bool   include_header = true; ///< Whether to emit the table header row.
};

/** Associates a phase label with its calculated summary. */
using LabeledSummary = std::pair<std::string, SummaryStats>;

/**
 * Formats an optional numeric value for console output.
 *
 * @param[in] value Value to format, or empty when unavailable.
 * @param[in] options Numeric formatting controls.
 * @return Formatted value, or `n/a` when @p value is empty.
 */
inline std::string format_optional(const std::optional<double>& value, const ConsoleSummaryOptions& options)
{
    if(!value.has_value())
    {
        return "n/a";
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(options.precision) << *value;
    return out.str();
}

/**
 * Formats aggregate statistics as a single console line.
 *
 * @param[in] summary Statistics to format.
 * @param[in] options Numeric formatting controls.
 * @return Console-ready summary text.
 */
inline std::string format_summary(const SummaryStats& summary, const ConsoleSummaryOptions& options = ConsoleSummaryOptions{})
{
    std::ostringstream out;
    out << "mean=" << format_optional(summary.mean, options)
        << " min=" << format_optional(summary.min, options)
        << " p90=" << format_optional(summary.p90, options)
        << " p99=" << format_optional(summary.p99, options)
        << " max=" << format_optional(summary.max, options);
    return out.str();
}

/**
 * Formats statistics for one named phase.
 *
 * @param[in] label Phase label.
 * @param[in] summary Statistics to format.
 * @param[in] options Numeric formatting controls.
 * @return Console-ready phase summary text.
 */
inline std::string format_phase_summary(const std::string& label, const SummaryStats& summary, const ConsoleSummaryOptions& options = ConsoleSummaryOptions{})
{
    std::ostringstream out;
    out << label << " "
        << format_optional(summary.mean, options)
        << " (" << format_optional(summary.min, options)
        << ", " << format_optional(summary.p90, options)
        << ", " << format_optional(summary.p99, options)
        << ", " << format_optional(summary.max, options)
        << ")";
    return out.str();
}

/**
 * Formats multiple phase summaries on one console line.
 *
 * @param[in] summaries Labeled statistics to format.
 * @param[in] options Numeric formatting controls.
 * @return Space-separated phase summaries.
 */
inline std::string format_phase_summaries(const std::vector<LabeledSummary>& summaries, const ConsoleSummaryOptions& options = ConsoleSummaryOptions{})
{
    std::ostringstream out;
    for(size_t idx = 0; idx < summaries.size(); ++idx)
    {
        if(idx != 0)
        {
            out << " ";
        }
        out << format_phase_summary(summaries[idx].first, summaries[idx].second, options);
    }
    return out.str();
}

/**
 * Formats an optional numeric table cell.
 *
 * @param[in] value Value to format, or empty when unavailable.
 * @param[in] options Numeric formatting controls.
 * @return Formatted value, or `n/a` when @p value is empty.
 */
inline std::string format_table_cell(const std::optional<double>& value, const ConsoleTableOptions& options)
{
    if(!value.has_value())
    {
        return "n/a";
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(options.precision) << *value;
    return out.str();
}

/**
 * Formats labeled statistics as a text table.
 *
 * @param[in] summaries Labeled statistics to format.
 * @param[in] options Table formatting controls.
 * @return Header and row strings for the formatted table.
 */
inline std::vector<std::string> format_phase_summary_table_lines(const std::vector<LabeledSummary>& summaries,
                                                                 const ConsoleTableOptions& options = ConsoleTableOptions{})
{
    std::vector<std::string> lines;
    if(options.include_header)
    {
        std::ostringstream header;
        header << std::left << std::setw(static_cast<int>(options.label_width)) << "Phase"
               << std::right
               << std::setw(static_cast<int>(options.numeric_width)) << "Mean"
               << std::setw(static_cast<int>(options.numeric_width)) << "Min"
               << std::setw(static_cast<int>(options.numeric_width)) << "P90"
               << std::setw(static_cast<int>(options.numeric_width)) << "P99"
               << std::setw(static_cast<int>(options.numeric_width)) << "Max";
        lines.push_back(header.str());
    }

    lines.reserve(lines.size() + summaries.size());
    for(const LabeledSummary& summary : summaries)
    {
        std::ostringstream row;
        row << std::left << std::setw(static_cast<int>(options.label_width)) << summary.first
            << std::right
            << std::setw(static_cast<int>(options.numeric_width)) << format_table_cell(summary.second.mean, options)
            << std::setw(static_cast<int>(options.numeric_width)) << format_table_cell(summary.second.min, options)
            << std::setw(static_cast<int>(options.numeric_width)) << format_table_cell(summary.second.p90, options)
            << std::setw(static_cast<int>(options.numeric_width)) << format_table_cell(summary.second.p99, options)
            << std::setw(static_cast<int>(options.numeric_width)) << format_table_cell(summary.second.max, options);
        lines.push_back(row.str());
    }

    return lines;
}

/** Collects one named sequence of finite timing or count samples. */
class SampleSeries
{
public:
    /**
     * Creates an empty sample sequence.
     *
     * @param[in] name Series name written to reports.
     * @param[in] unit Unit associated with every sample.
     */
    explicit SampleSeries(std::string name, TimeUnit unit = TimeUnit::Microseconds) :
        name_(std::move(name)),
        unit_(unit)
    {
    }

    /**
     * Returns the series name.
     *
     * @return Immutable series name.
     */
    const std::string& name() const { return name_; }
    /**
     * Returns the unit associated with the samples.
     *
     * @return Sample unit.
     */
    TimeUnit           unit() const { return unit_; }
    /**
     * Returns the recorded samples.
     *
     * @return Immutable sample sequence.
     */
    const std::vector<double>& samples() const { return samples_; }

    /**
     * Appends one finite sample.
     *
     * @param[in] sample_us Sample value in the unit associated with this series.
     * @throws std::invalid_argument if @p sample_us is not finite.
     */
    void add_sample(double sample_us)
    {
        if(!std::isfinite(sample_us))
        {
            throw std::invalid_argument("SampleSeries::add_sample requires a finite sample");
        }
        samples_.push_back(sample_us);
    }

    /**
     * Calculates descriptive statistics for the recorded samples.
     *
     * @return Calculated statistics.
     */
    SummaryStats summary() const
    {
        SummaryStats stats;
        stats.count = samples_.size();
        if(samples_.empty())
        {
            return stats;
        }

        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());

        const auto minmax = std::minmax_element(samples_.begin(), samples_.end());
        const double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
        const double mean = sum / static_cast<double>(samples_.size());

        double variance_sum = 0.0;
        for(double sample : samples_)
        {
            const double diff = sample - mean;
            variance_sum += diff * diff;
        }

        stats.min  = *minmax.first;
        stats.max  = *minmax.second;
        stats.mean = mean;
        stats.stddev = (samples_.size() > 1) ? std::sqrt(variance_sum / static_cast<double>(samples_.size() - 1)) : 0.0;
        stats.p50 = percentile(sorted, 0.50);
        stats.p90 = percentile(sorted, 0.90);
        stats.p95 = percentile(sorted, 0.95);
        stats.p99 = percentile(sorted, 0.99);
        return stats;
    }

    /**
     * Generates a histogram for the recorded samples.
     *
     * @param[in] config Histogram range and binning controls.
     * @return Generated histogram; empty when no samples are recorded.
     * @throws std::invalid_argument if the requested configuration is invalid.
     */
    Histogram histogram(const HistogramConfig& config = HistogramConfig{}) const
    {
        if(config.bin_count == 0)
        {
            throw std::invalid_argument("HistogramConfig::bin_count must be greater than zero");
        }

        Histogram histogram;
        if(samples_.empty())
        {
            return histogram;
        }

        const auto minmax = std::minmax_element(samples_.begin(), samples_.end());
        double range_min  = config.min.value_or(*minmax.first);
        double range_max  = config.max.value_or(*minmax.second);

        if(config.bin_width.has_value() && *config.bin_width <= 0.0)
        {
            throw std::invalid_argument("HistogramConfig::bin_width must be greater than zero");
        }

        size_t bin_count = config.bin_count;
        if(config.bin_width.has_value())
        {
            const double width = *config.bin_width;
            if(!config.max.has_value())
            {
                const double needed_width = std::max(0.0, *minmax.second - range_min);
                range_max = range_min + std::max(1.0, std::ceil(needed_width / width)) * width;
            }
            bin_count = static_cast<size_t>(std::ceil((range_max - range_min) / width));
            bin_count = std::max<size_t>(bin_count, 1);
        }
        else if(!config.min.has_value() && !config.max.has_value() && range_min == range_max)
        {
            range_min = std::max(0.0, range_min - 0.5);
            range_max = std::max(1.0, range_max + 0.5);
        }

        if(!(range_min < range_max))
        {
            throw std::invalid_argument("Histogram range requires min < max");
        }

        const double bin_width = (range_max - range_min) / static_cast<double>(bin_count);
        histogram.bins.reserve(bin_count);
        for(size_t bin_idx = 0; bin_idx < bin_count; ++bin_idx)
        {
            const double lower = range_min + static_cast<double>(bin_idx) * bin_width;
            const double upper = (bin_idx + 1 == bin_count) ? range_max : lower + bin_width;
            histogram.bins.push_back({lower, upper, 0});
        }

        for(double sample : samples_)
        {
            if(sample < range_min)
            {
                ++histogram.underflow;
            }
            else if(sample > range_max)
            {
                ++histogram.overflow;
            }
            else if(sample == range_max)
            {
                ++histogram.bins.back().count;
            }
            else
            {
                const auto bin_idx = static_cast<size_t>((sample - range_min) / bin_width);
                ++histogram.bins[std::min(bin_idx, histogram.bins.size() - 1)].count;
            }
        }

        return histogram;
    }

private:
    static double percentile(const std::vector<double>& sorted, double percentile_rank)
    {
        if(sorted.size() == 1)
        {
            return sorted.front();
        }

        const double scaled_index = percentile_rank * static_cast<double>(sorted.size() - 1);
        const auto   lower_idx    = static_cast<size_t>(std::floor(scaled_index));
        const auto   upper_idx    = static_cast<size_t>(std::ceil(scaled_index));
        const double fraction     = scaled_index - static_cast<double>(lower_idx);
        return sorted[lower_idx] + (sorted[upper_idx] - sorted[lower_idx]) * fraction;
    }

    std::string         name_;
    TimeUnit            unit_;
    std::vector<double> samples_;
};

/** Controls timing-report generation. */
struct ReportOptions
{
    HistogramConfig histogram_config; ///< Default binning used for report series.
};

/** Collects metadata and sample summaries for a JSON timing report. */
class TimingReport
{
public:
    /** Value types accepted for report metadata. */
    using MetadataValue = std::variant<std::string, int64_t, double, bool>;

    /**
     * Creates a report for one benchmark.
     *
     * @param[in] benchmark Benchmark name written to the report.
     */
    explicit TimingReport(std::string benchmark) :
        benchmark_(std::move(benchmark))
    {
    }

    /**
     * Adds string metadata.
     *
     * @param[in] key Metadata name.
     * @param[in] value Metadata value.
     */
    void add_metadata(std::string key, std::string value)
    {
        metadata_.push_back({std::move(key), std::move(value)});
    }

    /**
     * Adds C-string metadata.
     *
     * @param[in] key Metadata name.
     * @param[in] value Null-terminated metadata value.
     * @throws std::invalid_argument if @p value is null.
     */
    void add_metadata(std::string key, const char* value)
    {
        if(value == nullptr)
        {
            throw std::invalid_argument("Timing metadata C-string value must not be null");
        }
        add_metadata(std::move(key), std::string(value));
    }

    /**
     * Adds 64-bit integer metadata.
     *
     * @param[in] key Metadata name.
     * @param[in] value Metadata value.
     */
    void add_metadata(std::string key, int64_t value)
    {
        metadata_.push_back({std::move(key), value});
    }

    /**
     * Adds integer metadata.
     *
     * @param[in] key Metadata name.
     * @param[in] value Metadata value.
     */
    void add_metadata(std::string key, int value)
    {
        add_metadata(std::move(key), static_cast<int64_t>(value));
    }

    /**
     * Adds floating-point metadata.
     *
     * @param[in] key Metadata name.
     * @param[in] value Finite metadata value.
     * @throws std::invalid_argument if @p value is not finite.
     */
    void add_metadata(std::string key, double value)
    {
        if(!std::isfinite(value))
        {
            throw std::invalid_argument("TimingReport metadata values must be finite");
        }
        metadata_.push_back({std::move(key), value});
    }

    /**
     * Adds Boolean metadata.
     *
     * @param[in] key Metadata name.
     * @param[in] value Metadata value.
     */
    void add_metadata(std::string key, bool value)
    {
        metadata_.push_back({std::move(key), value});
    }

    /**
     * Calculates and adds one sample series to the report.
     *
     * @param[in] series Samples to summarize.
     * @param[in] tags Stable key-value tags associated with the series.
     * @param[in] histogram_config Histogram configuration for this series.
     * @throws std::invalid_argument if @p histogram_config is invalid for a nonempty series.
     */
    void add_series(const SampleSeries& series,
                    std::vector<std::pair<std::string, std::string>> tags = {},
                    const HistogramConfig& histogram_config = HistogramConfig{})
    {
        series_.push_back({series.name(),
                           unit_string(series.unit()),
                           std::move(tags),
                           series.summary(),
                           series.histogram(histogram_config)});
    }

    /**
     * Writes the report as JSON.
     *
     * @param[in] path Destination JSON file path.
     * @param[in] options Report-generation controls reserved for the output format.
     * @throws std::runtime_error if the file cannot be opened or written.
     */
    void write_json(const std::string& path, const ReportOptions&) const
    {
        std::ofstream out(path);
        if(!out)
        {
            throw std::runtime_error("Failed to open timing report JSON file: " + path);
        }

        out << "{\n";
        out << "  \"schema_version\": 1,\n";
        out << "  \"benchmark\": " << quote(benchmark_) << ",\n";
        out << "  \"metadata\": ";
        write_metadata(out, 2);
        out << ",\n";
        out << "  \"series\": [\n";
        for(size_t idx = 0; idx < series_.size(); ++idx)
        {
            write_series(out, series_[idx], 4);
            if(idx + 1 != series_.size())
            {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";

        if(!out)
        {
            throw std::runtime_error("Failed to write timing report JSON file: " + path);
        }
    }

private:
    struct SeriesEntry
    {
        std::string                                      name;
        std::string                                      unit;
        std::vector<std::pair<std::string, std::string>> tags;
        SummaryStats                                     summary;
        Histogram                                        histogram;
    };

    static std::string indent(size_t count)
    {
        return std::string(count, ' ');
    }

    static std::string quote(const std::string& value)
    {
        std::ostringstream out;
        out << '"';
        for(char ch : value)
        {
            switch(ch)
            {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if(static_cast<unsigned char>(ch) < 0x20)
                {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
                }
                else
                {
                    out << ch;
                }
                break;
            }
        }
        out << '"';
        return out.str();
    }

    static void write_number(std::ostream& out, double value)
    {
        out << std::setprecision(12) << value;
    }

    static void write_optional_number(std::ostream& out, const std::optional<double>& value)
    {
        if(value.has_value())
        {
            write_number(out, *value);
        }
        else
        {
            out << "null";
        }
    }

    static void write_metadata_value(std::ostream& out, const MetadataValue& value)
    {
        std::visit(
            [&out](const auto& item)
            {
                using item_t = std::decay_t<decltype(item)>;
                if constexpr(std::is_same_v<item_t, std::string>)
                {
                    out << quote(item);
                }
                else if constexpr(std::is_same_v<item_t, bool>)
                {
                    out << (item ? "true" : "false");
                }
                else if constexpr(std::is_same_v<item_t, double>)
                {
                    write_number(out, item);
                }
                else
                {
                    out << item;
                }
            },
            value);
    }

    void write_metadata(std::ostream& out, size_t base_indent) const
    {
        if(metadata_.empty())
        {
            out << "{}";
            return;
        }

        out << "{\n";
        for(size_t idx = 0; idx < metadata_.size(); ++idx)
        {
            out << indent(base_indent + 2) << quote(metadata_[idx].first) << ": ";
            write_metadata_value(out, metadata_[idx].second);
            if(idx + 1 != metadata_.size())
            {
                out << ",";
            }
            out << "\n";
        }
        out << indent(base_indent) << "}";
    }

    static void write_tags(std::ostream& out,
                           const std::vector<std::pair<std::string, std::string>>& tags,
                           size_t base_indent)
    {
        if(tags.empty())
        {
            out << "{}";
            return;
        }

        out << "{\n";
        for(size_t idx = 0; idx < tags.size(); ++idx)
        {
            out << indent(base_indent + 2) << quote(tags[idx].first) << ": " << quote(tags[idx].second);
            if(idx + 1 != tags.size())
            {
                out << ",";
            }
            out << "\n";
        }
        out << indent(base_indent) << "}";
    }

    static void write_summary(std::ostream& out, const SummaryStats& summary, size_t base_indent)
    {
        out << "{\n";
        out << indent(base_indent + 2) << "\"count\": " << summary.count << ",\n";
        out << indent(base_indent + 2) << "\"min\": ";
        write_optional_number(out, summary.min);
        out << ",\n";
        out << indent(base_indent + 2) << "\"mean\": ";
        write_optional_number(out, summary.mean);
        out << ",\n";
        out << indent(base_indent + 2) << "\"max\": ";
        write_optional_number(out, summary.max);
        out << ",\n";
        out << indent(base_indent + 2) << "\"stddev\": ";
        write_optional_number(out, summary.stddev);
        out << ",\n";
        out << indent(base_indent + 2) << "\"p50\": ";
        write_optional_number(out, summary.p50);
        out << ",\n";
        out << indent(base_indent + 2) << "\"p90\": ";
        write_optional_number(out, summary.p90);
        out << ",\n";
        out << indent(base_indent + 2) << "\"p95\": ";
        write_optional_number(out, summary.p95);
        out << ",\n";
        out << indent(base_indent + 2) << "\"p99\": ";
        write_optional_number(out, summary.p99);
        out << "\n";
        out << indent(base_indent) << "}";
    }

    static void write_histogram(std::ostream& out, const Histogram& histogram, size_t base_indent)
    {
        out << "{\n";
        out << indent(base_indent + 2) << "\"binning\": \"linear\",\n";
        out << indent(base_indent + 2) << "\"bin_count\": " << histogram.bins.size() << ",\n";
        out << indent(base_indent + 2) << "\"underflow\": " << histogram.underflow << ",\n";
        out << indent(base_indent + 2) << "\"overflow\": " << histogram.overflow << ",\n";
        out << indent(base_indent + 2) << "\"bins\": [\n";
        for(size_t idx = 0; idx < histogram.bins.size(); ++idx)
        {
            const HistogramBin& bin = histogram.bins[idx];
            out << indent(base_indent + 4) << "{\"lower\": ";
            write_number(out, bin.lower);
            out << ", \"upper\": ";
            write_number(out, bin.upper);
            out << ", \"count\": " << bin.count << "}";
            if(idx + 1 != histogram.bins.size())
            {
                out << ",";
            }
            out << "\n";
        }
        out << indent(base_indent + 2) << "]\n";
        out << indent(base_indent) << "}";
    }

    static void write_series(std::ostream& out, const SeriesEntry& series, size_t base_indent)
    {
        out << indent(base_indent) << "{\n";
        out << indent(base_indent + 2) << "\"name\": " << quote(series.name) << ",\n";
        out << indent(base_indent + 2) << "\"unit\": " << quote(series.unit) << ",\n";
        out << indent(base_indent + 2) << "\"tags\": ";
        write_tags(out, series.tags, base_indent + 2);
        out << ",\n";
        out << indent(base_indent + 2) << "\"summary\": ";
        write_summary(out, series.summary, base_indent + 2);
        out << ",\n";
        out << indent(base_indent + 2) << "\"histogram\": ";
        write_histogram(out, series.histogram, base_indent + 2);
        out << "\n";
        out << indent(base_indent) << "}";
    }

    std::string                                  benchmark_;
    std::vector<std::pair<std::string, MetadataValue>> metadata_;
    std::vector<SeriesEntry>                    series_;
};

} // namespace cuphy::examples::timing

#endif // !defined(CUPHY_EXAMPLES_TIMING_HPP_INCLUDED_)
