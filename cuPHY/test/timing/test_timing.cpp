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

#include "timing.hpp"

#include "gtest/gtest.h"

#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <unistd.h>

namespace
{
namespace timing = cuphy::examples::timing;

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

size_t total_bin_count(const timing::Histogram& histogram)
{
    return std::accumulate(histogram.bins.begin(),
                           histogram.bins.end(),
                           static_cast<size_t>(0),
                           [](size_t total, const timing::HistogramBin& bin) { return total + bin.count; });
}

std::filesystem::path unique_report_path()
{
    static std::atomic_uint64_t next_id{0};
    return std::filesystem::temp_directory_path() /
           ("cuphy_timing_report_test_" + std::to_string(getpid()) + "_" + std::to_string(next_id++) + ".json");
}
} // namespace

TEST(TimingSampleSeries, ComputesSummaryStatsAndInterpolatedPercentiles)
{
    timing::SampleSeries series("gpu.run");
    for(double sample : {1.0, 2.0, 3.0, 4.0, 5.0})
    {
        series.add_sample(sample);
    }

    const timing::SummaryStats summary = series.summary();

    EXPECT_EQ(summary.count, 5);
    ASSERT_TRUE(summary.min.has_value());
    ASSERT_TRUE(summary.max.has_value());
    ASSERT_TRUE(summary.mean.has_value());
    ASSERT_TRUE(summary.stddev.has_value());
    ASSERT_TRUE(summary.p50.has_value());
    ASSERT_TRUE(summary.p90.has_value());
    ASSERT_TRUE(summary.p95.has_value());
    ASSERT_TRUE(summary.p99.has_value());
    EXPECT_DOUBLE_EQ(*summary.min, 1.0);
    EXPECT_DOUBLE_EQ(*summary.max, 5.0);
    EXPECT_DOUBLE_EQ(*summary.mean, 3.0);
    EXPECT_NEAR(*summary.stddev, std::sqrt(2.5), 1.0e-12);
    EXPECT_DOUBLE_EQ(*summary.p50, 3.0);
    EXPECT_DOUBLE_EQ(*summary.p90, 4.6);
    EXPECT_DOUBLE_EQ(*summary.p95, 4.8);
    EXPECT_DOUBLE_EQ(*summary.p99, 4.96);
}

TEST(TimingSampleSeries, ComputesPercentilesForEvenSampleCounts)
{
    timing::SampleSeries series("cpu.setup");
    for(double sample : {10.0, 20.0, 30.0, 40.0})
    {
        series.add_sample(sample);
    }

    const timing::SummaryStats summary = series.summary();

    ASSERT_TRUE(summary.p50.has_value());
    ASSERT_TRUE(summary.p90.has_value());
    EXPECT_DOUBLE_EQ(*summary.p50, 25.0);
    EXPECT_DOUBLE_EQ(*summary.p90, 37.0);
}

TEST(TimingSampleSeries, AllowsEmptySeries)
{
    const timing::SampleSeries series("empty");
    const timing::SummaryStats summary = series.summary();

    EXPECT_EQ(summary.count, 0);
    EXPECT_FALSE(summary.min.has_value());
    EXPECT_FALSE(summary.max.has_value());
    EXPECT_FALSE(summary.mean.has_value());
    EXPECT_FALSE(summary.stddev.has_value());
    EXPECT_FALSE(summary.p50.has_value());
    EXPECT_FALSE(summary.p90.has_value());
    EXPECT_FALSE(summary.p95.has_value());
    EXPECT_FALSE(summary.p99.has_value());
}

TEST(TimingSampleSeries, RejectsNonFiniteSamples)
{
    timing::SampleSeries series("bad");

    EXPECT_THROW(series.add_sample(std::numeric_limits<double>::infinity()), std::invalid_argument);
    EXPECT_THROW(series.add_sample(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST(TimingHistogram, ComputesAutoLinearBinsWithFinalInclusiveBin)
{
    timing::SampleSeries series("gpu.run");
    for(double sample : {0.0, 1.0, 2.0, 3.0, 4.0})
    {
        series.add_sample(sample);
    }

    timing::HistogramConfig config;
    config.bin_count = 4;
    const timing::Histogram histogram = series.histogram(config);

    ASSERT_EQ(histogram.bins.size(), 4);
    EXPECT_EQ(histogram.underflow, 0);
    EXPECT_EQ(histogram.overflow, 0);
    EXPECT_DOUBLE_EQ(histogram.bins[0].lower, 0.0);
    EXPECT_DOUBLE_EQ(histogram.bins[0].upper, 1.0);
    EXPECT_DOUBLE_EQ(histogram.bins[3].lower, 3.0);
    EXPECT_DOUBLE_EQ(histogram.bins[3].upper, 4.0);
    EXPECT_EQ(histogram.bins[0].count, 1);
    EXPECT_EQ(histogram.bins[1].count, 1);
    EXPECT_EQ(histogram.bins[2].count, 1);
    EXPECT_EQ(histogram.bins[3].count, 2);
}

TEST(TimingHistogram, HonorsConfiguredRangeWithUnderflowAndOverflow)
{
    timing::SampleSeries series("gpu.run");
    for(double sample : {-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0})
    {
        series.add_sample(sample);
    }

    timing::HistogramConfig config;
    config.bin_count = 2;
    config.min = 0.0;
    config.max = 2.0;
    const timing::Histogram histogram = series.histogram(config);

    ASSERT_EQ(histogram.bins.size(), 2);
    EXPECT_EQ(histogram.underflow, 1);
    EXPECT_EQ(histogram.overflow, 1);
    EXPECT_EQ(histogram.bins[0].count, 2);
    EXPECT_EQ(histogram.bins[1].count, 3);
}

TEST(TimingHistogram, HandlesIdenticalSamples)
{
    timing::SampleSeries series("gpu.run");
    for(double sample : {5.0, 5.0, 5.0, 5.0})
    {
        series.add_sample(sample);
    }

    timing::HistogramConfig config;
    config.bin_count = 4;
    const timing::Histogram histogram = series.histogram(config);

    ASSERT_EQ(histogram.bins.size(), 4);
    EXPECT_EQ(histogram.underflow, 0);
    EXPECT_EQ(histogram.overflow, 0);
    EXPECT_DOUBLE_EQ(histogram.bins.front().lower, 4.5);
    EXPECT_DOUBLE_EQ(histogram.bins.back().upper, 5.5);
    EXPECT_EQ(total_bin_count(histogram), 4);
}

TEST(TimingReport, WritesJsonSummaryAndHistogram)
{
    timing::SampleSeries series("gpu.run");
    for(double sample : {1.0, 2.0, 3.0})
    {
        series.add_sample(sample);
    }

    timing::TimingReport report("unit_test");
    report.add_metadata("iterations", int64_t{3});
    report.add_metadata("input", std::string{"example.h5"});
    report.add_metadata("ref_check", true);
    report.add_metadata("snr", 10.5);
    report.add_series(series, {{"backend", "cuda_event"}, {"phase", "run"}});

    const std::filesystem::path output_path = unique_report_path();
    report.write_json(output_path.string(), timing::ReportOptions{});

    const std::string json = read_file(output_path);
    std::filesystem::remove(output_path);

    EXPECT_NE(json.find("\"schema_version\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"benchmark\": \"unit_test\""), std::string::npos);
    EXPECT_NE(json.find("\"iterations\": 3"), std::string::npos);
    EXPECT_NE(json.find("\"input\": \"example.h5\""), std::string::npos);
    EXPECT_NE(json.find("\"ref_check\": true"), std::string::npos);
    EXPECT_NE(json.find("\"snr\": 10.5"), std::string::npos);
    EXPECT_NE(json.find("\"name\": \"gpu.run\""), std::string::npos);
    EXPECT_NE(json.find("\"unit\": \"us\""), std::string::npos);
    EXPECT_NE(json.find("\"backend\": \"cuda_event\""), std::string::npos);
    EXPECT_NE(json.find("\"p90\": 2.8"), std::string::npos);
    EXPECT_NE(json.find("\"histogram\""), std::string::npos);
    EXPECT_NE(json.find("\"bins\""), std::string::npos);
}

TEST(TimingReport, RejectsNonFiniteDoubleMetadata)
{
    timing::TimingReport report("unit_test");

    EXPECT_THROW(report.add_metadata("nan", std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(report.add_metadata("infinity", std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(TimingReport, RejectsNullCStringMetadata)
{
    timing::TimingReport report("unit_test");

    EXPECT_THROW(report.add_metadata("input", static_cast<const char*>(nullptr)), std::invalid_argument);
}

TEST(TimingConsoleFormatting, FormatsSummaryWithDefaultPrecisionAndOrder)
{
    timing::SampleSeries series("gpu.run");
    for(double sample : {1.0, 2.0, 3.0, 4.0, 5.0})
    {
        series.add_sample(sample);
    }

    EXPECT_EQ(timing::format_summary(series.summary()), "mean=3.000 min=1.000 p90=4.600 p99=4.960 max=5.000");
}

TEST(TimingConsoleFormatting, FormatsPhaseSummariesWithLabels)
{
    timing::SampleSeries first("first");
    timing::SampleSeries second("second");
    for(double sample : {1.0, 2.0, 3.0})
    {
        first.add_sample(sample);
        second.add_sample(sample + 10.0);
    }

    const std::string formatted = timing::format_phase_summaries({
        {"Run-P1", first.summary()},
        {"Setup-P1", second.summary()}
    });

    EXPECT_EQ(formatted,
              "Run-P1 2.000 (1.000, 2.800, 2.980, 3.000) "
              "Setup-P1 12.000 (11.000, 12.800, 12.980, 13.000)");
}

TEST(TimingConsoleFormatting, FormatsEmptySummaryAsNotAvailable)
{
    const timing::SampleSeries series("empty");

    EXPECT_EQ(timing::format_summary(series.summary()), "mean=n/a min=n/a p90=n/a p99=n/a max=n/a");
}

TEST(TimingConsoleFormatting, FormatsPhaseSummaryTableLines)
{
    timing::SampleSeries first("first");
    timing::SampleSeries second("second");
    for(double sample : {1.0, 2.0, 3.0})
    {
        first.add_sample(sample);
        second.add_sample(sample + 10.0);
    }

    timing::ConsoleTableOptions options;
    options.label_width = 10;
    options.numeric_width = 8;

    const std::vector<std::string> lines = timing::format_phase_summary_table_lines({
        {"Run-P1", first.summary()},
        {"Setup-P1", second.summary()}
    }, options);

    ASSERT_EQ(lines.size(), 3);
    EXPECT_EQ(lines[0], "Phase         Mean     Min     P90     P99     Max");
    EXPECT_EQ(lines[1], "Run-P1       2.000   1.000   2.800   2.980   3.000");
    EXPECT_EQ(lines[2], "Setup-P1    12.000  11.000  12.800  12.980  13.000");
}

TEST(TimingCpuTimer, ThrowsWhenStoppedBeforeStart)
{
    timing::CpuTimer timer;
    EXPECT_THROW(static_cast<void>(timer.stop_us()), std::logic_error);
}
