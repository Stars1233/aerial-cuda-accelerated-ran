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

#ifndef NV_SCOPE_EXIT_HPP_INCLUDED_
#define NV_SCOPE_EXIT_HPP_INCLUDED_

#include <utility>

namespace nv
{

// RAII guard for scoped cleanup:
// - Runs cleanup on scope exit by default (including early returns/errors).
// - Call release() to disarm when ownership/cleanup transfers successfully.
// - Copy is disabled to prevent accidental double cleanup.
template <typename F>
class ScopeExit
{
public:
    explicit ScopeExit(F&& fn) : fn_(std::forward<F>(fn)), active_(true) {}
    ~ScopeExit() { if (active_) { fn_(); } }

    void release() { active_ = false; }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&& other) noexcept : fn_(std::move(other.fn_)), active_(other.active_) { other.active_ = false; }
    ScopeExit& operator=(ScopeExit&&) = delete;

private:
    F fn_;
    bool active_;
};

template <typename F>
ScopeExit<F> make_scope_exit(F&& fn)
{
    return ScopeExit<F>(std::forward<F>(fn));
}

} // namespace nv

#endif // NV_SCOPE_EXIT_HPP_INCLUDED_

