// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "aerial/containers/static_vector.hpp"

// aerial_common propagates gsl_CONFIG_CONTRACT_VIOLATION_THROWS via its INTERFACE compile
// definitions. If this fires, the CMakeLists.txt INTERFACE propagation is broken.
#ifndef gsl_CONFIG_CONTRACT_VIOLATION_THROWS
#error "aerial_common must propagate gsl_CONFIG_CONTRACT_VIOLATION_THROWS to all consumers"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gsl-lite/gsl-lite.hpp>
#include <gtest/gtest.h>

namespace {

struct LifetimeCounter final
{
    static inline int live_count = 0;
    static inline int construction_count = 0;
    static inline int destruction_count = 0;

    int value{0};

    LifetimeCounter()
    {
        ++live_count;
        ++construction_count;
    }

    explicit LifetimeCounter(const int value_in) : value(value_in)
    {
        ++live_count;
        ++construction_count;
    }

    LifetimeCounter(const LifetimeCounter &other) : value(other.value)
    {
        ++live_count;
        ++construction_count;
    }

    LifetimeCounter(LifetimeCounter &&other) noexcept : value(other.value)
    {
        other.value = -1;
        ++live_count;
        ++construction_count;
    }

    LifetimeCounter &operator=(const LifetimeCounter &) = default;
    LifetimeCounter &operator=(LifetimeCounter &&) noexcept = default;

    ~LifetimeCounter() noexcept
    {
        --live_count;
        ++destruction_count;
    }

    static void reset() noexcept
    {
        live_count = 0;
        construction_count = 0;
        destruction_count = 0;
    }
};

struct alignas(64) OverAlignedValue final
{
    int value{0};

    explicit OverAlignedValue(const int value_in) : value(value_in) {}
};

struct ThrowingElement final
{
    static inline int live_count = 0;
    static inline int construction_count = 0;
    static inline int destruction_count = 0;
    static inline int construction_attempt = 0;
    static inline int throw_on_attempt = 0;

    int value{0};

    ThrowingElement()
    {
        begin_construction();
    }

    explicit ThrowingElement(const int value_in) : value(value_in)
    {
        begin_construction();
    }

    ThrowingElement(const ThrowingElement &other) : value(other.value)
    {
        begin_construction();
    }

    ThrowingElement(ThrowingElement &&other) noexcept(false) : value(other.value)
    {
        other.value = -1;
        begin_construction();
    }

    ThrowingElement &operator=(const ThrowingElement &) = default;
    ThrowingElement &operator=(ThrowingElement &&) noexcept(false) = default;

    ~ThrowingElement() noexcept
    {
        --live_count;
        ++destruction_count;
    }

    static void reset() noexcept
    {
        live_count = 0;
        construction_count = 0;
        destruction_count = 0;
        construction_attempt = 0;
        throw_on_attempt = 0;
    }

    static void reset_attempts() noexcept
    {
        construction_attempt = 0;
        throw_on_attempt = 0;
    }

private:
    static void begin_construction()
    {
        ++construction_attempt;
        if ((throw_on_attempt != 0) && (construction_attempt == throw_on_attempt))
        {
            throw std::runtime_error{"ThrowingElement construction failed"};
        }
        ++live_count;
        ++construction_count;
    }
};

using ThrowingVector = aerial::static_vector<ThrowingElement, 4>;

static_assert(std::is_same_v<aerial::static_vector<int, 4>::value_type, int>);
static_assert(aerial::static_vector<int, 4>::capacity() == 4U);
static_assert(aerial::static_vector<int, 4>::max_size() == 4U);

TEST(StaticVector, DefaultConstructionIsEmpty)
{
    aerial::static_vector<int, 4> values;

    EXPECT_TRUE(values.empty());
    EXPECT_FALSE(values.full());
    EXPECT_EQ(values.size(), 0U);
    EXPECT_EQ(values.capacity(), 4U);
    EXPECT_EQ(values.begin(), values.end());
}

TEST(StaticVector, ZeroCapacityIsSupported)
{
    aerial::static_vector<int, 0> values;

    EXPECT_TRUE(values.empty());
    EXPECT_TRUE(values.full());
    EXPECT_EQ(values.size(), 0U);
    EXPECT_EQ(values.capacity(), 0U);
    EXPECT_EQ(values.data(), nullptr);
    EXPECT_THROW(static_cast<void>(values.emplace_back(1)), gsl_lite::fail_fast);
}

TEST(StaticVector, PushBackAndElementAccess)
{
    aerial::static_vector<int, 3> values;

    values.push_back(10);
    values.push_back(20);
    values.push_back(30);

    ASSERT_TRUE(values.full());
    EXPECT_EQ(values.front(), 10);
    EXPECT_EQ(values.back(), 30);
    EXPECT_EQ(values[1], 20);

    values[1] = 22;
    EXPECT_EQ(values[1], 22);
}

TEST(StaticVector, ConstAccess)
{
    const aerial::static_vector<int, 3> values{1, 2, 3};

    EXPECT_EQ(values.front(), 1);
    EXPECT_EQ(values.back(), 3);
    EXPECT_EQ(values[2], 3);
    EXPECT_EQ(values.cbegin(), values.begin());
    EXPECT_EQ(values.cend(), values.end());
}

TEST(StaticVector, EmplaceBackConstructsInPlace)
{
    aerial::static_vector<LifetimeCounter, 2> values;

    auto &first = values.emplace_back(7);
    auto &second = values.emplace_back(11);

    EXPECT_EQ(first.value, 7);
    EXPECT_EQ(second.value, 11);
    EXPECT_EQ(values.size(), 2U);
}

TEST(StaticVector, EmplaceBackReturnsLiveObjectInEmbeddedStorage)
{
    aerial::static_vector<OverAlignedValue, 2> values;

    auto &first = values.emplace_back(7);
    auto &second = values.emplace_back(11);

    EXPECT_EQ(&first, values.data());
    EXPECT_EQ(&second, values.data() + 1);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&first) % alignof(OverAlignedValue), 0U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&second) % alignof(OverAlignedValue), 0U);

    first.value = 17;
    second.value = 23;
    EXPECT_EQ(values[0].value, 17);
    EXPECT_EQ(values[1].value, 23);
}

TEST(StaticVector, PopBackAndClearDestroyActiveElements)
{
    LifetimeCounter::reset();

    {
        aerial::static_vector<LifetimeCounter, 4> values;
        static_cast<void>(values.emplace_back(1));
        static_cast<void>(values.emplace_back(2));
        static_cast<void>(values.emplace_back(3));

        EXPECT_EQ(LifetimeCounter::live_count, 3);
        values.pop_back();
        EXPECT_EQ(values.size(), 2U);
        EXPECT_EQ(LifetimeCounter::live_count, 2);

        values.clear();
        EXPECT_TRUE(values.empty());
        EXPECT_EQ(LifetimeCounter::live_count, 0);
    }

    EXPECT_EQ(LifetimeCounter::live_count, 0);
    EXPECT_EQ(LifetimeCounter::construction_count, LifetimeCounter::destruction_count);
}

TEST(StaticVector, ResizeGrowsAndShrinks)
{
    aerial::static_vector<int, 5> values;

    values.resize(3);
    ASSERT_EQ(values.size(), 3U);
    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[2], 0);

    values.resize(5, 9);
    ASSERT_EQ(values.size(), 5U);
    EXPECT_EQ(values[3], 9);
    EXPECT_EQ(values[4], 9);

    values.resize(1);
    ASSERT_EQ(values.size(), 1U);
    EXPECT_EQ(values[0], 0);
}

TEST(StaticVector, CountAndInitializerConstructors)
{
    aerial::static_vector<int, 4> repeated(3, 8);
    EXPECT_EQ(repeated.size(), 3U);
    EXPECT_EQ(repeated[0], 8);
    EXPECT_EQ(repeated[2], 8);

    aerial::static_vector<int, 4> initialized{1, 2, 3};
    EXPECT_EQ(initialized.size(), 3U);
    EXPECT_EQ(initialized[0], 1);
    EXPECT_EQ(initialized[2], 3);
}

TEST(StaticVector, CountConstructorCleansUpWhenDefaultConstructionThrows)
{
    ThrowingElement::reset();
    ThrowingElement::throw_on_attempt = 3;

    EXPECT_THROW(static_cast<void>(ThrowingVector(4)), std::runtime_error);

    EXPECT_EQ(ThrowingElement::live_count, 0);
    EXPECT_EQ(ThrowingElement::construction_count, ThrowingElement::destruction_count);
}

TEST(StaticVector, CountValueConstructorCleansUpWhenCopyConstructionThrows)
{
    ThrowingElement::reset();

    {
        const ThrowingElement value{9};
        ThrowingElement::reset_attempts();
        ThrowingElement::throw_on_attempt = 3;

        EXPECT_THROW(static_cast<void>(ThrowingVector(4, value)), std::runtime_error);

        EXPECT_EQ(ThrowingElement::live_count, 1);
        EXPECT_EQ(ThrowingElement::construction_count - ThrowingElement::destruction_count, 1);
    }

    EXPECT_EQ(ThrowingElement::live_count, 0);
    EXPECT_EQ(ThrowingElement::construction_count, ThrowingElement::destruction_count);
}

TEST(StaticVector, InitializerListConstructorCleansUpWhenCopyConstructionThrows)
{
    ThrowingElement::reset();
    ThrowingElement::throw_on_attempt = 7;

    EXPECT_THROW(static_cast<void>(ThrowingVector{ThrowingElement{1}, ThrowingElement{2},
                                                 ThrowingElement{3}, ThrowingElement{4}}),
                 std::runtime_error);

    EXPECT_EQ(ThrowingElement::live_count, 0);
    EXPECT_EQ(ThrowingElement::construction_count, ThrowingElement::destruction_count);
}

TEST(StaticVector, CopyConstructorCleansUpWhenCopyConstructionThrows)
{
    ThrowingElement::reset();

    {
        aerial::static_vector<ThrowingElement, 4> source{ThrowingElement{1}, ThrowingElement{2},
                                                        ThrowingElement{3}, ThrowingElement{4}};
        ThrowingElement::reset_attempts();
        ThrowingElement::throw_on_attempt = 3;

        EXPECT_THROW(
                [&source]() {
                    ThrowingVector copy{source};
                    static_cast<void>(copy);
                }(),
                std::runtime_error);

        EXPECT_EQ(ThrowingElement::live_count, static_cast<int>(source.size()));
        EXPECT_EQ(ThrowingElement::construction_count - ThrowingElement::destruction_count,
                  static_cast<int>(source.size()));
    }

    EXPECT_EQ(ThrowingElement::live_count, 0);
    EXPECT_EQ(ThrowingElement::construction_count, ThrowingElement::destruction_count);
}

TEST(StaticVector, MoveConstructorCleansUpWhenMoveConstructionThrows)
{
    ThrowingElement::reset();

    {
        aerial::static_vector<ThrowingElement, 4> source{ThrowingElement{1}, ThrowingElement{2},
                                                        ThrowingElement{3}, ThrowingElement{4}};
        ThrowingElement::reset_attempts();
        ThrowingElement::throw_on_attempt = 3;

        EXPECT_THROW(
                [&source]() {
                    ThrowingVector moved{std::move(source)};
                    static_cast<void>(moved);
                }(),
                std::runtime_error);

        EXPECT_EQ(ThrowingElement::live_count, static_cast<int>(source.size()));
        EXPECT_EQ(ThrowingElement::construction_count - ThrowingElement::destruction_count,
                  static_cast<int>(source.size()));
    }

    EXPECT_EQ(ThrowingElement::live_count, 0);
    EXPECT_EQ(ThrowingElement::construction_count, ThrowingElement::destruction_count);
}

TEST(StaticVector, MoveOnlyValues)
{
    aerial::static_vector<std::unique_ptr<int>, 3> values;

    static_cast<void>(values.emplace_back(std::make_unique<int>(7)));
    values.push_back(std::make_unique<int>(9));

    ASSERT_EQ(values.size(), 2U);
    ASSERT_NE(values[0], nullptr);
    ASSERT_NE(values[1], nullptr);
    EXPECT_EQ(*values[0], 7);
    EXPECT_EQ(*values[1], 9);
}

TEST(StaticVector, MoveConstructionPreservesSourceSize)
{
    aerial::static_vector<std::unique_ptr<int>, 2> source;
    source.push_back(std::make_unique<int>(1));
    source.push_back(std::make_unique<int>(2));

    aerial::static_vector<std::unique_ptr<int>, 2> moved(std::move(source));

    EXPECT_EQ(source.size(), 2U);
    ASSERT_EQ(moved.size(), 2U);
    ASSERT_NE(moved[0], nullptr);
    ASSERT_NE(moved[1], nullptr);
    EXPECT_EQ(*moved[0], 1);
    EXPECT_EQ(*moved[1], 2);
}

TEST(StaticVector, StorageIsContiguous)
{
    aerial::static_vector<int, 4> values{3, 4, 5};

    ASSERT_EQ(values.size(), 3U);
    EXPECT_EQ(values.data(), &values[0]);
    EXPECT_EQ(values.begin() + 1, values.data() + 1);
    EXPECT_EQ(*(values.begin() + 2), 5);
}

TEST(StaticVector, IteratesWithStandardAlgorithms)
{
    aerial::static_vector<int, 4> values{1, 2, 3, 4};

    EXPECT_EQ(std::accumulate(values.begin(), values.end(), 0), 10);
    EXPECT_EQ(*values.rbegin(), 4);
    EXPECT_EQ(*values.crbegin(), 4);
}

TEST(StaticVector, ContractFailuresThrowFailFast)
{
    aerial::static_vector<int, 2> values;

    EXPECT_THROW(values.pop_back(), gsl_lite::fail_fast);
    EXPECT_THROW(static_cast<void>(values[0]), gsl_lite::fail_fast);
    EXPECT_THROW(values.resize(3), gsl_lite::fail_fast);

    values.push_back(1);
    values.push_back(2);
    EXPECT_THROW(values.push_back(3), gsl_lite::fail_fast);
    EXPECT_THROW(static_cast<void>(values.emplace_back(4)), gsl_lite::fail_fast);
    EXPECT_THROW((aerial::static_vector<int, 1>{1, 2}), gsl_lite::fail_fast);
}

} // namespace
