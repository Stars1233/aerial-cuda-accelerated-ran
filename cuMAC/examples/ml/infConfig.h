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

#pragma once

// Lightweight, dependency-free reader for the trtEngine inference benchmark
// configuration (infConfig.yaml). It supports the small subset of YAML used by
// that file: top-level scalars, one level of nested sections, '#' comments, and
// optionally quoted string values. Keys are addressed with dotted notation,
// e.g. "engine.useCudaGraph". This avoids adding a yaml-cpp build dependency.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "trtEngine.h"

namespace cumac_ml {

class YamlConfig {
public:
    // Parse the given file. Returns false if it cannot be opened.
    bool load(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            return false;
        }
        m_path = path;
        m_values.clear();

        std::string line;
        std::string section;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            const size_t firstNonSpace = line.find_first_not_of(" \t");
            if (firstNonSpace == std::string::npos) {
                continue;  // blank line
            }
            if (line[firstNonSpace] == '#') {
                continue;  // full-line comment
            }

            // Strip a trailing inline comment (" #..."). Values containing '#'
            // are not expected in this config.
            const size_t commentPos = line.find(" #");
            if (commentPos != std::string::npos) {
                line = line.substr(0, commentPos);
            }

            const bool indented = (firstNonSpace > 0);
            const std::string content = trim(line);
            if (content.empty()) {
                continue;
            }

            const size_t colon = content.find(':');
            if (colon == std::string::npos) {
                continue;  // not a key: value line
            }
            const std::string key = trim(content.substr(0, colon));
            const std::string rawValue = trim(content.substr(colon + 1));

            if (!indented) {
                if (rawValue.empty()) {
                    section = key;  // top-level section header
                    continue;
                }
                section.clear();
                m_values[key] = unquote(rawValue);
            } else {
                const std::string fullKey = section.empty() ? key : (section + "." + key);
                m_values[fullKey] = unquote(rawValue);
            }
        }
        return true;
    }

    bool has(const std::string& key) const {
        return m_values.find(key) != m_values.end();
    }

    std::string getString(const std::string& key, const std::string& def) const {
        auto it = m_values.find(key);
        return it == m_values.end() ? def : it->second;
    }

    long getInt(const std::string& key, long def) const {
        auto it = m_values.find(key);
        if (it == m_values.end() || it->second.empty()) {
            return def;
        }
        try {
            return std::stol(it->second);
        } catch (...) {
            return def;
        }
    }

    double getDouble(const std::string& key, double def) const {
        auto it = m_values.find(key);
        if (it == m_values.end() || it->second.empty()) {
            return def;
        }
        try {
            return std::stod(it->second);
        } catch (...) {
            return def;
        }
    }

    bool getBool(const std::string& key, bool def) const {
        auto it = m_values.find(key);
        if (it == m_values.end() || it->second.empty()) {
            return def;
        }
        std::string v = toLower(it->second);
        if (v == "true" || v == "1" || v == "yes" || v == "on") {
            return true;
        }
        if (v == "false" || v == "0" || v == "no" || v == "off") {
            return false;
        }
        return def;
    }

    const std::string& path() const { return m_path; }

    static std::string toLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

private:
    static std::string trim(const std::string& s) {
        const size_t b = s.find_first_not_of(" \t");
        if (b == std::string::npos) {
            return "";
        }
        const size_t e = s.find_last_not_of(" \t");
        return s.substr(b, e - b + 1);
    }

    static std::string unquote(const std::string& s) {
        if (s.size() >= 2 &&
            ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
            return s.substr(1, s.size() - 2);
        }
        return s;
    }

    std::map<std::string, std::string> m_values;
    std::string m_path;
};


// Aggregated configuration for the trtEngine latency benchmark, combining the
// engine build/run options (trtEngineConfig) with harness/runtime options.
struct infConfig {
    trtEngineConfig engine;        // model source, precision, CUDA Graph, caching, .pt conversion.
    int  gpuId             = 0;    // CUDA device index.
    int  batchSize         = 16;   // Inference batch size (number of UEs per forward pass).
    int  maxBatchSize      = 16;   // Max batch for the dynamic optimization profile.
    int  obsDim            = 487;  // Input feature dimension.
    int  actionDim         = 28;   // Output dimension (MCS-offset logits).
    int  warmupIters       = 200;  // Warmup iterations before timing (also primes the CUDA Graph).
    int  timingIters       = 2000; // Measured iterations.
    bool copyInputEachIter = false;// Include the H2D input copy in the timed loop.
    std::string configPath;        // Path the config was loaded from.
};


// Resolve a possibly-relative path against the configuration file's directory.
inline std::string resolveRelativeTo(const std::string& p, const std::filesystem::path& baseDir) {
    if (p.empty()) {
        return p;
    }
    std::filesystem::path pp(p);
    if (pp.is_absolute()) {
        return pp.string();
    }
    return (baseDir / pp).lexically_normal().string();
}


// Load and validate an infConfig.yaml. Throws std::runtime_error on failure.
inline infConfig loadInfConfig(const std::string& path) {
    YamlConfig yaml;
    if (!yaml.load(path)) {
        throw std::runtime_error("infConfig: cannot open config file: " + path);
    }

    infConfig cfg;
    cfg.configPath = path;
    const std::filesystem::path baseDir = std::filesystem::path(path).parent_path();

    // --- model ---
    const std::string modelPath = yaml.getString("model.path", "");
    if (modelPath.empty()) {
        throw std::runtime_error("infConfig: 'model.path' is required in " + path);
    }
    cfg.engine.modelPath            = resolveRelativeTo(modelPath, baseDir);
    cfg.engine.inputName            = yaml.getString("model.inputName", "obs");
    cfg.engine.outputName           = yaml.getString("model.outputName", "logits");
    cfg.engine.includeObsNormalizer = yaml.getBool("model.includeObsNormalizer", false);
    cfg.obsDim                      = static_cast<int>(yaml.getInt("model.obsDim", 487));
    cfg.actionDim                   = static_cast<int>(yaml.getInt("model.actionDim", 28));

    // --- precision (top-level scalar: fp32 | fp16) ---
    const std::string prec = YamlConfig::toLower(yaml.getString("precision", "fp32"));
    cfg.engine.precision = (prec == "fp16" || prec == "half") ? trtPrecision::kFP16
                                                              : trtPrecision::kFP32;

    // --- engine ---
    cfg.engine.useCudaGraph      = yaml.getBool("engine.useCudaGraph", false);
    cfg.engine.workspaceMiB      = static_cast<size_t>(yaml.getInt("engine.workspaceMiB", 1024));
    cfg.engine.builderOptimizationLevel =
        static_cast<int32_t>(yaml.getInt("engine.builderOptimizationLevel", 5));
    cfg.engine.enableEngineCache = yaml.getBool("engine.cache", true);
    const std::string cacheDir   = yaml.getString("engine.cacheDir", "");
    cfg.engine.engineCacheDir    = cacheDir.empty() ? std::string() : resolveRelativeTo(cacheDir, baseDir);
    cfg.engine.forceRebuild      = yaml.getBool("engine.forceRebuild", false);

    // --- conversion (.pt -> ONNX) ---
    cfg.engine.pythonExecutable  = yaml.getString("conversion.python", "python3");
    const std::string script     = yaml.getString("conversion.script", "pt_to_onnx.py");
    cfg.engine.ptToOnnxScript    = resolveRelativeTo(script, baseDir);

    // --- runtime ---
    cfg.gpuId        = static_cast<int>(yaml.getInt("runtime.gpuId", 0));
    cfg.batchSize    = static_cast<int>(yaml.getInt("runtime.batchSize", 16));
    cfg.maxBatchSize = static_cast<int>(yaml.getInt("runtime.maxBatchSize", cfg.batchSize));
    if (cfg.maxBatchSize < cfg.batchSize) {
        cfg.maxBatchSize = cfg.batchSize;
    }

    // --- latency ---
    cfg.warmupIters       = static_cast<int>(yaml.getInt("latency.warmupIters", 200));
    cfg.timingIters       = static_cast<int>(yaml.getInt("latency.timingIters", 2000));
    cfg.copyInputEachIter = yaml.getBool("latency.copyInputEachIter", false);

    cfg.engine.verbose = yaml.getBool("verbose", true);

    // Sane fallbacks for the I/O dimensions.
    if (cfg.obsDim <= 0)    cfg.obsDim = 487;
    if (cfg.actionDim <= 0) cfg.actionDim = 28;

    return cfg;
}

}  // namespace cumac_ml
