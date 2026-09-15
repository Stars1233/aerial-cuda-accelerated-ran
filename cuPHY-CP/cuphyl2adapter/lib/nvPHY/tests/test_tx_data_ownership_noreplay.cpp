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

#include <gtest/gtest.h>

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace
{

/**
 * @brief Parent directory of a filesystem path (test helper).
 *
 * @param[in] path  NUL-terminated filesystem path.
 * @return Substring up to (excluding) the last '/', or empty if none is present.
 */
std::string dirname_of(const char* path)
{
    const std::string s(path);
    const auto        pos = s.find_last_of('/');
    return pos == std::string::npos ? std::string{} : s.substr(0, pos);
}

/**
 * @brief Strip C/C++ comments so the structural regex guards below cannot match
 *        commented-out code or documentation mentions of set_deferred(..., true).
 *
 * A minimal state machine: `//` line and `/`+`*` block comments are removed while
 * string and char literals are preserved (so `//` or `/`+`*` inside a literal is
 * not mistaken for a comment). Newlines are kept so line/function-structure
 * matching (extract_fn_body, the #ifdef ... #endif regex) is unaffected.
 *
 * @param[in] src  Source text to scan.
 * @return @p src with comments removed and string/char literals and newlines preserved.
 */
std::string strip_comments(const std::string& src)
{
    std::string out;
    out.reserve(src.size());
    enum class State
    {
        Code,
        LineComment,
        BlockComment,
        StringLit,
        CharLit,
    };
    State st = State::Code;
    for(std::size_t i = 0; i < src.size(); ++i)
    {
        const char c = src[i];
        const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
        switch(st)
        {
        case State::Code:
            if(c == '/' && n == '/') { st = State::LineComment; ++i; }
            else if(c == '/' && n == '*') { st = State::BlockComment; ++i; }
            else if(c == '"') { st = State::StringLit; out += c; }
            else if(c == '\'') { st = State::CharLit; out += c; }
            else { out += c; }
            break;
        case State::LineComment:
            if(c == '\n') { st = State::Code; out += c; }
            break;
        case State::BlockComment:
            if(c == '*' && n == '/') { st = State::Code; ++i; }
            else if(c == '\n') { out += c; }
            break;
        case State::StringLit:
            out += c;
            if(c == '\\' && n != '\0') { out += n; ++i; }
            else if(c == '"') { st = State::Code; }
            break;
        case State::CharLit:
            out += c;
            if(c == '\\' && n != '\0') { out += n; ++i; }
            else if(c == '\'') { st = State::Code; }
            break;
        }
    }
    return out;
}

/**
 * @brief Load a production nvPHY source file (comments stripped) for static structural guards.
 *
 * @param[in] basename  File name relative to the nvPHY source dir
 *                      (e.g. "nv_phy_slot_dispatch.cpp").
 * @return Comment-stripped file contents, or empty (after adding a test failure)
 *         if none of the candidate paths resolve.
 */
std::string read_source_file(const char* basename)
{
    const std::string test_dir = dirname_of(__FILE__);
    // Resolve from test source dir first (works when run from any cwd), then
    // common build-tree layouts.
    const std::vector<std::string> candidates = {
        test_dir + "/../" + basename,
        std::string("../../../../../../cuPHY-CP/cuphyl2adapter/lib/nvPHY/") + basename,
        std::string("../../../cuPHY-CP/cuphyl2adapter/lib/nvPHY/") + basename,
        std::string("cuPHY-CP/cuphyl2adapter/lib/nvPHY/") + basename,
    };
    for(const auto& path : candidates)
    {
        std::ifstream in(path);
        if(in)
        {
            std::ostringstream ss;
            ss << in.rdbuf();
            return strip_comments(ss.str());
        }
    }
    ADD_FAILURE() << "unable to open source file: " << basename;
    return {};
}

/**
 * @brief Count non-overlapping regex matches in @p haystack.
 *
 * @param[in] haystack  Text to search.
 * @param[in] pattern   Compiled regex to count.
 * @return Number of non-overlapping matches.
 */
std::size_t count_regex_matches(const std::string& haystack, const std::regex& pattern)
{
    const auto begin = std::sregex_iterator(haystack.begin(), haystack.end(), pattern);
    const auto end   = std::sregex_iterator();
    return static_cast<std::size_t>(std::distance(begin, end));
}

/**
 * @brief Extract a PHY_module member function body from production source text.
 *
 * @param[in] src      Full (comment-stripped) source text.
 * @param[in] fn_name  Bare member name, e.g. "launch_tx_data_h2d".
 * @return Text from the "void PHY_module::<fn_name>(" definition up to the next
 *         such definition (or end of file), or empty if the definition is absent.
 */
std::string extract_fn_body(const std::string& src, const std::string& fn_name)
{
    const std::string needle = "void PHY_module::" + fn_name + "(";
    const auto        start  = src.find(needle);
    if(start == std::string::npos)
    {
        return {};
    }
    const auto next = src.find("\nvoid PHY_module::", start + needle.size());
    const auto len  = (next == std::string::npos ? src.size() : next) - start;
    return src.substr(start, len);
}

} // namespace

// This guard reads production source text (not build-flavor-specific compiled
// code), so it enforces the structural invariant that holds for BOTH replay and
// no-replay builds:
//   - slot_dispatch has exactly one set_deferred(..., true): inside launch_tx_data_h2d.
//   - slot_dispatch has zero set_deferred(..., true) inside enqueue_channel_tasks.
//   - nv_phy_module has exactly one set_deferred(..., true), and it is wrapped in
//     an #ifdef ENABLE_FAPI_STORE_REPLAY region so no-replay builds never compile it.
TEST(TxDataOwnership, ArmSitesMatchBuildFlavor)
{
    const std::string dispatch_src = read_source_file("nv_phy_slot_dispatch.cpp");
    const std::string module_src   = read_source_file("nv_phy_module.cpp");
    ASSERT_FALSE(dispatch_src.empty());
    ASSERT_FALSE(module_src.empty());

    // `;`-bounded, nested-paren-safe: matches set_deferred(foo(x), true) but not
    // any set_deferred(..., false) statement.
    const std::regex arm_true_re(R"(set_deferred\s*\([^;]*?,\s*true\s*\))");

    // Exactly one arm in slot_dispatch, and it must be inside launch_tx_data_h2d.
    EXPECT_EQ(count_regex_matches(dispatch_src, arm_true_re), 1U)
        << "slot_dispatch must contain exactly one set_deferred(true) site";
    EXPECT_EQ(count_regex_matches(extract_fn_body(dispatch_src, "launch_tx_data_h2d"), arm_true_re), 1U)
        << "the sole slot_dispatch arm must be inside launch_tx_data_h2d";

    // enqueue_channel_tasks must never arm (guards against GT-12162-style regressions).
    EXPECT_EQ(count_regex_matches(extract_fn_body(dispatch_src, "enqueue_channel_tasks"), arm_true_re), 0U)
        << "enqueue_channel_tasks must not arm deferred";

    // Exactly one arm in nv_phy_module (the replay-present direct-mode site).
    EXPECT_EQ(count_regex_matches(module_src, arm_true_re), 1U)
        << "nv_phy_module must contain exactly one set_deferred(true) site";

    // That module arm must be gated by ENABLE_FAPI_STORE_REPLAY and by
    // fapi_to_cplane_direct so no-replay builds never compile it.
    EXPECT_EQ(count_regex_matches(
                  module_src,
                  std::regex(
                      R"(#ifdef ENABLE_FAPI_STORE_REPLAY[\s\S]{0,4000}?fapi_to_cplane_direct[\s\S]{0,800}?set_deferred\s*\([^;]*?,\s*true\s*\)[\s\S]{0,400}?#endif)")),
              1U)
        << "the nv_phy_module arm must be inside an #ifdef ENABLE_FAPI_STORE_REPLAY block gated on fapi_to_cplane_direct";
}
